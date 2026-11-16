from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from gpu4agents.core.config import GatewayConfig, RuntimeConfig
from gpu4agents.gateway import EndpointSource, GatewayServer, adapter_names, get_adapter
from gpu4agents.gateway.base import Upstream
from gpu4agents.gateway import service as svc

from conftest import BASE_CONFIG

UP = Upstream("http://1.2.3.4:40301", "http://1.2.3.4:40301/v1", "SECRETKEY", "served/Model-27B")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --- adapters --------------------------------------------------------------

def test_builtin_adapters_registered():
    assert adapter_names() == ["anthropic", "anthropic-openai", "none", "openai"]


def test_anthropic_drops_client_creds_and_injects_key():
    cfg = GatewayConfig(adapter="anthropic")
    adapter = get_adapter(cfg)
    headers = {"Authorization": "Bearer OAUTH-LEAK", "x-api-key": "leak",
               "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    out_h, _ = adapter.rewrite_request(dict(headers), b"", UP)
    assert out_h["Authorization"] == "Bearer SECRETKEY"
    assert out_h["x-api-key"] == "SECRETKEY"
    assert "OAUTH-LEAK" not in json.dumps(out_h)
    assert out_h["anthropic-version"] == "2023-06-01"  # harmless header preserved


def test_model_rewrite_and_effort_normalization():
    cfg = GatewayConfig(
        adapter="anthropic",
        reasoning_effort={"map": {"high": "xhigh"}, "supported": ["xhigh", "medium", "low"],
                          "on_unsupported": "drop"},
    )
    adapter = get_adapter(cfg)
    body = json.dumps({"model": "claude-opus-4-8", "reasoning_effort": "high",
                       "reasoning": {"effort": "high"}}).encode()
    _, out = adapter.rewrite_request({"Content-Type": "application/json"}, body, UP)
    obj = json.loads(out)
    assert obj["model"] == "served/Model-27B"
    assert obj["reasoning_effort"] == "xhigh"
    assert obj["reasoning"]["effort"] == "xhigh"


def test_effort_in_output_config_is_normalized():
    # Claude Code sends the effort as output_config.effort, not reasoning_effort.
    cfg = GatewayConfig(adapter="anthropic",
                        reasoning_effort={"map": {"high": "xhigh"}, "supported": ["xhigh", "medium", "low"],
                                          "on_unsupported": "drop"})
    adapter = get_adapter(cfg)
    body = json.dumps({"output_config": {"effort": "high"},
                       "thinking": {"type": "adaptive"}}).encode()
    _, out = adapter.rewrite_request({"Content-Type": "application/json"}, body, UP)
    assert json.loads(out)["output_config"]["effort"] == "xhigh"


def test_effort_unsupported_is_dropped():
    cfg = GatewayConfig(adapter="openai",
                        reasoning_effort={"supported": ["xhigh", "medium", "low"], "on_unsupported": "drop"})
    adapter = get_adapter(cfg)
    _, out = adapter.rewrite_request({"Content-Type": "application/json"},
                                     json.dumps({"reasoning_effort": "ultra"}).encode(), UP)
    assert "reasoning_effort" not in json.loads(out)


def test_effort_default_is_noop_keeps_standard_high():
    # default policy must not break a standard OpenAI server that accepts "high"
    adapter = get_adapter(GatewayConfig(adapter="openai"))
    _, out = adapter.rewrite_request({"Content-Type": "application/json"},
                                     json.dumps({"reasoning_effort": "high"}).encode(), UP)
    assert json.loads(out)["reasoning_effort"] == "high"


def test_effort_error_policy_raises():
    cfg = GatewayConfig(adapter="openai",
                        reasoning_effort={"supported": ["low"], "on_unsupported": "error"})
    adapter = get_adapter(cfg)
    with pytest.raises(ValueError):
        adapter.rewrite_request({"Content-Type": "application/json"},
                                json.dumps({"reasoning_effort": "high"}).encode(), UP)


def test_none_adapter_is_passthrough():
    adapter = get_adapter(GatewayConfig(adapter="none"))
    h, b = adapter.rewrite_request({"Authorization": "Bearer keep"}, b"x", UP)
    assert h == {"Authorization": "Bearer keep"} and b == b"x"


def test_unknown_adapter_errors():
    from gpu4agents.core.errors import JambuError
    with pytest.raises(JambuError):
        get_adapter(GatewayConfig(adapter="nope"))


# --- config ----------------------------------------------------------------

def test_config_defaults_gateway_off():
    rc = RuntimeConfig.model_validate(BASE_CONFIG)
    assert rc.gateway.enabled is False
    assert rc.gateway.adapter == "openai"


def test_config_parses_gateway_section():
    body = dict(BASE_CONFIG)
    body["gateway"] = {"enabled": True, "adapter": "anthropic", "listen": "127.0.0.1:18080"}
    rc = RuntimeConfig.model_validate(body)
    assert rc.gateway.listen_host_port == ("127.0.0.1", 18080)


def test_bad_listen_rejected():
    with pytest.raises(ValueError):
        GatewayConfig(listen="not-a-port").listen_host_port


# --- service (systemd unit) ------------------------------------------------

def test_service_unit_is_restarting_and_boot_safe(tmp_path):
    unit = svc.render_unit(project_dir=tmp_path, python="/usr/bin/python3")
    assert "Type=simple" in unit
    assert "Restart=always" in unit
    assert "WantedBy=default.target" in unit
    assert f"WorkingDirectory={tmp_path}" in unit
    assert "gpu4agents" in unit and "gateway" in unit


def test_service_slug_is_stable_and_per_project(tmp_path):
    a = svc.unit_name(tmp_path / "proj-a")
    b = svc.unit_name(tmp_path / "proj-b")
    assert a != b
    assert a == svc.unit_name(tmp_path / "proj-a")  # deterministic


# --- end to end: forward + rewrite + follow the endpoint -------------------

class _Echo(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        payload = json.dumps({
            "seen_path": self.path,
            "seen_auth": self.headers.get("Authorization"),
            "seen_body": body.decode(),
            "tag": self.server.tag,  # type: ignore[attr-defined]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _start_upstream(tag: str):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    httpd.tag = tag  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_gateway_forwards_rewrites_and_follows_endpoint():
    up1, url1 = _start_upstream("first")
    up2, url2 = _start_upstream("second")
    current = {"endpoint": url1, "api_key": "KEY1", "model": "real/Model"}

    cfg = GatewayConfig(adapter="anthropic", listen=f"127.0.0.1:{_free_port()}",
                        reasoning_effort={"map": {"high": "xhigh"}})
    source = EndpointSource(lambda: dict(current, base_url=current["endpoint"] + "/v1"), ttl=0.0)
    server = GatewayServer(cfg, get_adapter(cfg), source)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.2)
    base = f"http://{cfg.listen}"
    try:
        r = httpx.post(f"{base}/v1/messages",
                       headers={"Authorization": "Bearer OAUTH-LEAK", "Content-Type": "application/json"},
                       content=json.dumps({"model": "claude-x", "reasoning_effort": "high"}).encode(),
                       timeout=10)
        seen = r.json()
        assert seen["tag"] == "first"
        assert seen["seen_path"] == "/v1/messages"
        assert seen["seen_auth"] == "Bearer KEY1"           # injected instance key
        body = json.loads(seen["seen_body"])
        assert body["model"] == "real/Model"                # model rewritten
        assert body["reasoning_effort"] == "xhigh"          # effort normalized

        # re-rent: endpoint changes; ttl=0 means every request re-reads it
        current["endpoint"] = url2
        current["api_key"] = "KEY2"
        r2 = httpx.post(f"{base}/v1/messages",
                        headers={"Content-Type": "application/json"},
                        content=json.dumps({"model": "m"}).encode(), timeout=10)
        assert r2.json()["tag"] == "second"                 # gateway followed the machine
    finally:
        server.stop()
        up1.shutdown()
        up2.shutdown()


# --- Task 3: adapters can rewrite the path and translate the response -------

from gpu4agents.gateway.base import GatewayAdapter  # noqa: E402


class _PathAndBodyAdapter(GatewayAdapter):
    """Rewrites /a -> /b and prefixes the streamed response body."""

    name = "stub"

    def rewrite_request(self, headers, body, upstream):
        return headers, body

    def target_path(self, path):
        return "/b" if path == "/a" else path

    def translate_response(self, request_body, status, headers, chunks):
        def gen():
            yield b"TRANSLATED:"
            for c in chunks:
                yield c
        return status, {**headers, "Content-Type": "text/plain"}, gen()


def test_default_adapter_is_response_passthrough():
    # the built-in anthropic adapter must not touch the path or the body
    a = get_adapter(GatewayConfig(adapter="anthropic"))
    assert a.target_path("/v1/messages") == "/v1/messages"
    status, headers, chunks = a.translate_response(b"{}", 200, {"X": "y"}, iter([b"hello"]))
    assert status == 200 and headers == {"X": "y"} and b"".join(chunks) == b"hello"


def test_gateway_lets_adapter_rewrite_path_and_translate_body():
    up, url = _start_upstream("echo")
    current = {"endpoint": url, "api_key": "K", "model": "m"}
    cfg = GatewayConfig(adapter="none", listen=f"127.0.0.1:{_free_port()}")
    source = EndpointSource(lambda: dict(current, base_url=url + "/v1"), ttl=0.0)
    server = GatewayServer(cfg, _PathAndBodyAdapter(cfg), source)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.2)
    try:
        r = httpx.post(f"http://{cfg.listen}/a",
                       headers={"Content-Type": "application/json"},
                       content=b'{"x":1}', timeout=10)
        assert r.content.startswith(b"TRANSLATED:")   # response body was translated
        seen = json.loads(r.content[len(b"TRANSLATED:"):])
        assert seen["seen_path"] == "/b"              # path was rewritten /a -> /b
        assert seen["tag"] == "echo"
    finally:
        server.stop()
        up.shutdown()


# --- Task 6: end to end - serve Anthropic over a fake OpenAI backend --------

def _anthropic_events(raw: bytes):
    out = []
    for block in raw.split(b"\n\n"):
        block = block.strip()
        if not block:
            continue
        name = data = None
        for line in block.split(b"\n"):
            if line.startswith(b"event:"):
                name = line[6:].strip().decode()
            elif line.startswith(b"data:"):
                data = json.loads(line[5:].strip())
        out.append((name, data))
    return out


class _OpenAIChat(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        self.server.seen_path = self.path          # type: ignore[attr-defined]
        self.server.seen_body = body               # type: ignore[attr-defined]
        if body.get("stream"):
            sse = (
                b'data: {"id":"x","model":"m","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
                b'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                b'data: {"id":"x","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1}}\n\n'
                b'data: [DONE]\n\n'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)
        else:
            payload = json.dumps({
                "id": "chatcmpl-x", "model": "m",
                "choices": [{"message": {"role": "assistant", "content": "Hi there"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def _start_openai_chat():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIChat)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _run_gateway(adapter_name):
    up, url = _start_openai_chat()
    cfg = GatewayConfig(adapter=adapter_name, listen=f"127.0.0.1:{_free_port()}")
    source = EndpointSource(lambda: {"endpoint": url, "base_url": url + "/v1",
                                     "api_key": "KEY", "model": "served/Model"}, ttl=0.0)
    server = GatewayServer(cfg, get_adapter(cfg), source)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.2)
    return up, server, cfg


def test_e2e_anthropic_over_openai_nonstreaming():
    up, server, cfg = _run_gateway("anthropic-openai")
    try:
        r = httpx.post(f"http://{cfg.listen}/v1/messages",
                       headers={"x-api-key": "LEAK", "Content-Type": "application/json"},
                       content=json.dumps({"model": "claude-x", "max_tokens": 16,
                                           "messages": [{"role": "user", "content": "hi"}]}).encode(),
                       timeout=10)
        msg = r.json()
        assert msg["type"] == "message"
        assert msg["content"] == [{"type": "text", "text": "Hi there"}]
        assert msg["usage"] == {"input_tokens": 4, "output_tokens": 2}
        # the backend saw an OpenAI chat request at the rewritten path
        assert up.seen_path == "/v1/chat/completions"
        assert up.seen_body["model"] == "served/Model"      # renamed
        assert up.seen_body["messages"][-1] == {"role": "user", "content": "hi"}
    finally:
        server.stop()
        up.shutdown()


def test_e2e_anthropic_over_openai_streaming():
    up, server, cfg = _run_gateway("anthropic-openai")
    try:
        with httpx.stream("POST", f"http://{cfg.listen}/v1/messages",
                          headers={"Content-Type": "application/json"},
                          content=json.dumps({"model": "claude-x", "max_tokens": 16, "stream": True,
                                              "messages": [{"role": "user", "content": "hi"}]}).encode(),
                          timeout=10) as resp:
            raw = b"".join(resp.iter_raw())
        events = _anthropic_events(raw)
        names = [n for n, _ in events]
        assert names[0] == "message_start" and names[-1] == "message_stop"
        text = "".join(d["delta"]["text"] for n, d in events
                        if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
        assert text == "Hi"
        assert up.seen_body["stream"] is True
    finally:
        server.stop()
        up.shutdown()
