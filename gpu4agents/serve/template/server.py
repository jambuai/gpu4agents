"""Your model server (`runtime.engine: custom`). Contract: SERVING.md.

gpu uploads this directory to the GPU machine and runs `run:` from gpujob.yaml
after `setup:`. Keep these four things and change the rest:

1. Listen on 0.0.0.0:$PORT.
2. GET /health answers 503 until the model is loaded, then 200.
3. Every other route needs `Authorization: Bearer $JAMBU_API_KEY`.
4. GET /metrics counts requests (jambu_requests_running, jambu_requests_total):
   the machine counts them as activity, so it stays up while clients use it
   and stops by itself after lifecycle.idle_timeout once they stop.

Write `load()` and the handlers in ROUTES. Use OpenAI's request and response
shapes where one exists (/v1/chat/completions, /v1/images/generations,
/v1/audio/speech, /v1/embeddings, ...) so any OpenAI client works unchanged.
"""

import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = os.environ.get("JAMBU_MODEL_ID", "")
API_KEY = os.environ.get("JAMBU_API_KEY", "")
PORT = int(os.environ.get("PORT") or 8000)

STATE = {"ready": False, "running": 0, "total": 0}
COUNTERS = threading.Lock()
GPU = threading.Lock()  # one request on the GPU at a time
MODEL = {}


def load():
    """Load the model once, before /health turns 200. Raise to fail the start."""
    # Example:
    #   import torch
    #   from diffusers import DiffusionPipeline
    #   MODEL["pipe"] = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    MODEL["name"] = MODEL_ID or "echo"


def echo(body):
    """Replace with your own handler: JSON in, JSON out."""
    return {"model": MODEL["name"], "echo": body}


ROUTES = {"/v1/echo": echo}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003
        return

    def _send(self, status, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else (
            json.dumps(payload).encode() if content_type == "application/json" else str(payload).encode()
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        return not API_KEY or self.headers.get("Authorization") == "Bearer " + API_KEY

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(200 if STATE["ready"] else 503, {"ready": STATE["ready"]})
        elif path == "/metrics":
            self._send(
                200,
                "jambu_requests_running %d\njambu_requests_total %d\n" % (STATE["running"], STATE["total"]),
                "text/plain",
            )
        elif not self._authorized():
            self._send(401, {"error": {"message": "missing or wrong API key"}})
        elif path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL["name"], "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "no route GET " + path}})

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        # Read the body before any reply, or a kept-alive connection desyncs.
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if not self._authorized():
            self._send(401, {"error": {"message": "missing or wrong API key"}})
            return
        handler = ROUTES.get(path)
        if handler is None:
            self._send(404, {"error": {"message": "no route POST " + path}})
            return
        if not STATE["ready"]:
            self._send(503, {"error": {"message": "the model is still loading"}})
            return
        with COUNTERS:
            STATE["running"] += 1
            STATE["total"] += 1
        try:
            body = json.loads(raw or b"{}")
            with GPU:
                self._send(200, handler(body))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": {"message": "%s: %s" % (type(exc).__name__, exc)}})
        finally:
            with COUNTERS:
                STATE["running"] -= 1


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True

    def _load():
        started = time.time()
        try:
            load()
        except Exception:  # noqa: BLE001 - a failed load ends the service; gpu shows this log
            traceback.print_exc()
            os._exit(1)
        STATE["ready"] = True
        print("server ready in %.0fs on port %d" % (time.time() - started, PORT), flush=True)

    threading.Thread(target=_load, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
