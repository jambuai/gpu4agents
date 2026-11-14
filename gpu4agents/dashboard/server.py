"""The dashboard's loopback HTTP server.

A stdlib ``ThreadingHTTPServer`` that serves a read-only JSON API
(``/api/state``) and a self-contained HTML page (``/``). It never mutates
anything and never loads a provider credential; it assembles its state from an
injected provider plus the reachability watch. The token gate guards every
route when a token is set.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from ..core.config import DashboardConfig
from .auth import TokenGate
from .probe import ReachabilityWatch

_LOOPBACK = {"127.0.0.1", "localhost", "::1", ""}
# The self-contained page, read once at import; no external resources.
_PAGE = (Path(__file__).parent / "page.html").read_text(encoding="utf-8")


def guard_bind(host: str, token: Optional[str], *, required: bool) -> None:
    """Refuse to serve the fleet open: a non-loopback bind needs a token.

    Raises ValueError when ``host`` is not loopback, ``required`` is set, and no
    token is configured. Loopback binds and token-protected binds are allowed.
    """
    if required and host not in _LOOPBACK and token is None:
        raise ValueError(
            f"refusing to bind {host!r} without a token: it would serve the fleet "
            "open to the network. Bind 127.0.0.1 and reach it over Tailscale/SSH, "
            "or set dashboard.options.token."
        )


def build_state(provider: Callable[[], dict], watch: ReachabilityWatch, now: float) -> dict[str, Any]:
    """Merge the provider's rows with the reachability watch into the API shape."""
    base = provider() or {}
    projects = list(base.get("projects") or [])
    instances = list(base.get("instances") or [])
    proxies = list(base.get("proxies") or [])
    probes: dict[str, Any] = {}
    for row in projects + instances + proxies:
        endpoint = row.get("endpoint") if isinstance(row, dict) else None
        if endpoint:
            snap = watch.snapshot(endpoint)
            probes[endpoint] = snap
            if isinstance(row, dict):
                row["reachable"] = snap.get("last_ok")
    return {
        "instances": instances,
        "projects": projects,
        "proxies": proxies,
        "probes": probes,
        "generated_at": now,
    }


_CONTROL_PREFIX = "/api/control/"


def make_handler(provider, watch, gate: TokenGate, log: Callable[[str], None],
                 control_enabled: bool = False, actions: Optional[dict] = None) -> type[BaseHTTPRequestHandler]:
    actions = actions or {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "gpu4agents-dashboard"

        def log_message(self, *_args):  # silence default stderr spam
            pass

        def _headers_dict(self) -> dict:
            return {k: v for k, v in self.headers.items()}

        def _send(self, code: int, body: bytes, ctype: str, extra: Optional[list] = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for name, value in extra or []:
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def _unauthorized(self) -> None:
            self._send(401, b'{"error":"unauthorized"}', "application/json")

        def do_HEAD(self) -> None:  # cheap liveness probe
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            path, query = parts.path, parts.query
            if not gate.check(self._headers_dict(), query):
                return self._unauthorized()
            # a fresh ?token= hand-off sets the cookie so a phone stays logged in
            extra = []
            token = TokenGate._from_query(query)
            if token is not None and gate.token is not None:
                extra.append(("Set-Cookie", gate.cookie_header(token)))
            if path == "/api/state":
                state = build_state(provider, watch, time.time())
                state["control"] = control_enabled
                log(f"GET /api/state -> 200 ({len(state.get('projects', []))} projects)")
                return self._send(200, json.dumps(state).encode(), "application/json", extra)
            if path == "/":
                return self._send(200, _PAGE.encode(), "text/html; charset=utf-8", extra)
            self._send(404, b'{"error":"not found"}', "application/json")

        def do_POST(self) -> None:
            parts = urlsplit(self.path)
            path, query = parts.path, parts.query
            # Mutating routes exist only when control is enabled; otherwise 404.
            if not (control_enabled and path.startswith(_CONTROL_PREFIX)):
                return self._send(404, b'{"error":"not found"}', "application/json")
            if not gate.check(self._headers_dict(), query):
                return self._unauthorized()
            action = path[len(_CONTROL_PREFIX):]
            if action not in actions:
                return self._send(404, b'{"error":"unknown action"}', "application/json")
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._send(400, b'{"error":"bad json"}', "application/json")
            if not (isinstance(body, dict) and body.get("confirm") is True):
                return self._send(400, b'{"error":"confirm required"}', "application/json")
            project = body.get("project")
            log(f"POST /api/control/{action} project={project!r} -> running")
            try:
                result = actions[action](project)
            except Exception as exc:  # noqa: BLE001 - surface the action's failure as 500
                payload = json.dumps({"error": str(exc)}).encode()
                return self._send(500, payload, "application/json")
            payload = json.dumps({"ok": True, "action": action, "project": project, "result": result}).encode()
            self._send(200, payload, "application/json")

    return Handler


class DashboardServer:
    def __init__(self, config: DashboardConfig, provider: Callable[[], dict],
                 watch: ReachabilityWatch, gate: TokenGate,
                 log: Callable[[str], None] = lambda _m: None,
                 actions: Optional[dict] = None) -> None:
        host, port = config.listen_host_port
        guard_bind(host, gate.token, required=config.token_required_off_loopback)
        self._log = log
        handler = make_handler(provider, watch, gate, log,
                               control_enabled=config.control_enabled, actions=actions)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._httpd.daemon_threads = True
        self.host, self.port = host, port

    def serve_forever(self) -> None:
        self._log(f"dashboard listening on http://{self.host}:{self.port}")
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._httpd.server_close()

    def stop(self) -> None:
        self._httpd.shutdown()
