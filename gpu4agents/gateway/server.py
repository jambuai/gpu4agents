"""The loopback HTTP server behind ``gpu gateway``.

A stdlib ``ThreadingHTTPServer`` bound to 127.0.0.1 that forwards every request
to the rented machine's current endpoint, applying the selected adapter first.
It re-reads the endpoint from state (via an injected provider) so it keeps
working when the machine is re-rented at a new host:port, and refreshes on a
connection error. Responses stream straight through (chunked), so SSE and large
bodies are untouched.

httpx is the only non-stdlib dependency, and it is already required by the CLI.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

import httpx

from ..core.config import GatewayConfig
from .base import GatewayAdapter, Upstream

# Hop-by-hop headers (RFC 7230) plus ones we always recompute. Never forwarded.
HOP = {
    "connection", "keep-alive", "transfer-encoding", "upgrade", "proxy-connection",
    "te", "trailer", "host", "content-length", "accept-encoding",
}


class EndpointSource:
    """Caches the current upstream; refreshes after ``ttl`` or when forced."""

    def __init__(self, provider: Callable[[], dict], ttl: float = 60.0) -> None:
        self._provider = provider
        self._ttl = ttl
        self._lock = threading.Lock()
        self._at = 0.0
        self._up: Optional[Upstream] = None

    def get(self, force: bool = False) -> Optional[Upstream]:
        with self._lock:
            if not force and self._up and (time.time() - self._at) < self._ttl:
                return self._up
            try:
                conn = self._provider()
            except Exception:  # noqa: BLE001 - never let a state read kill the server
                conn = None
            self._at = time.time()
            self._up = Upstream.from_connection(conn) if conn and conn.get("endpoint") else None
            return self._up


def make_handler(config: GatewayConfig, adapter: GatewayAdapter, source: EndpointSource,
                 log: Callable[[str], None]) -> type[BaseHTTPRequestHandler]:

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "gpu4agents-gateway"

        def log_message(self, *_args):  # silence the default stderr spam
            pass

        def _error(self, code: int, message: str) -> None:
            body = (f'{{"error":{{"type":"gateway_error","message":{message!r}}}}}').encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def _proxy(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            last_err: Optional[Exception] = None
            for attempt in range(2):
                up = source.get(force=attempt > 0)
                if up is None:
                    return self._error(503, "no endpoint; run `gpu setup` first")
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                headers, out = adapter.rewrite_request(headers, body, up)
                url = up.endpoint.rstrip("/") + self.path
                try:
                    with httpx.Client(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
                        with client.stream(self.command, url, content=out, headers=headers) as resp:
                            log(f"{self.command} {self.path} -> {resp.status_code}")
                            self.send_response(resp.status_code)
                            for k, v in resp.headers.items():
                                if k.lower() not in HOP:
                                    self.send_header(k, v)
                            self.send_header("Transfer-Encoding", "chunked")
                            self.send_header("Connection", "close")
                            self.end_headers()
                            for chunk in resp.iter_raw(65536):
                                if chunk:
                                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                                    self.wfile.flush()
                            self.wfile.write(b"0\r\n\r\n")
                    self.close_connection = True
                    return
                except httpx.ConnectError as exc:
                    last_err = exc
                    log(f"{self.command} {self.path} -> connect error; refreshing endpoint")
                    continue
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                    return
            self._error(503, f"upstream unreachable ({last_err}); it may have stopped when idle")

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _proxy

        def do_HEAD(self) -> None:  # cheap liveness probe for `gateway status`
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

    return Handler


class GatewayServer:
    """Owns the ThreadingHTTPServer; ``serve_forever`` blocks."""

    def __init__(self, config: GatewayConfig, adapter: GatewayAdapter, source: EndpointSource,
                 log: Callable[[str], None] = lambda _m: None) -> None:
        host, port = config.listen_host_port
        self._log = log
        self._adapter_name = adapter.name
        handler = make_handler(config, adapter, source, log)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._httpd.daemon_threads = True
        self.host, self.port = host, port

    def serve_forever(self) -> None:
        self._log(f"gateway listening on http://{self.host}:{self.port} (adapter={self._adapter_name})")
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._httpd.server_close()

    def stop(self) -> None:
        """Ask ``serve_forever`` to return (safe to call from another thread)."""
        self._httpd.shutdown()
