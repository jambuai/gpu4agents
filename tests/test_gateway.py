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
    assert adapter_names() == ["anthropic", "none", "openai"]


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
