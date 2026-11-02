"""Full-loop integration, with no external API involved.

The fake provider does not fake the boot: it **executes the real onstart script
the engine generated**, with absolute paths rewritten into a tmp dir. So this
exercises the actual chain

    engine.setup -> generated boot script -> real watchdog agent
                 -> health wait -> run + heartbeats -> idle guard -> provider action

which is everything between `gpu setup` and the GPU switching itself off.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fakes import FakeProvider

from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.engine import Engine
from gpu4agents.core.models import Endpoint, Instance, InstanceState
from gpu4agents.providers.base import WatchdogSpec

MARKER_ACTIONS = '''
import json as _json
import os as _os


def _record(action, ctx):
    with open(_os.environ["JAMBU_TEST_MARKER"], "w") as handle:
        _json.dump({"action": action, "instance_id": ctx.get("instance_id")}, handle)


def provider_stop(ctx):
    _record("stop", ctx)


def provider_destroy(ctx):
    _record("destroy", ctx)
'''


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        code = 200 if self.server.healthy else 503  # type: ignore[attr-defined]
        self.send_response(code)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *_args):  # noqa: A003
        return


@pytest.fixture
def model_server():
    """Stands in for the vLLM server the image would start."""
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _HealthHandler)
    server.healthy = False  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, port
    server.shutdown()


class BootingProvider(FakeProvider):
    """A provider that really runs the boot script the engine handed it."""

    def __init__(self, config, credentials, root: Path, marker: Path, model_port: int):
        super().__init__(config, credentials)
        self.root = root
        self.marker = marker
        self.model_port = model_port
        self.watchdog_port = config.lifecycle.watchdog.port
        self.boot_log = root / "boot.log"

    def watchdog_spec(self, spec_or_instance=None) -> WatchdogSpec:
        return WatchdogSpec(
            actions_source=MARKER_ACTIONS,
            env={"JAMBU_TEST_MARKER": str(self.marker)},
            context={"provider": "booting-fake"},
        )

    def setup(self, spec):
        instance = super().setup(spec)
        self._boot(spec.onstart)
        instance.endpoints = {
            "model": Endpoint(
                "model", f"http://127.0.0.1:{self.model_port}", 8000, self.model_port
            ),
            "watchdog": Endpoint(
                "watchdog",
                f"http://127.0.0.1:{self.watchdog_port}",
                self.watchdog_port,
                self.watchdog_port,
            ),
        }
        self.instances[instance.id] = instance
        return instance

    def _boot(self, onstart: str) -> None:
        rewritten = onstart
        for remote, local in (
            ("/etc/jambu", self.root / "etc"),
            ("/var/lib/jambu", self.root / "lib"),
            ("/var/log/jambu", self.root / "log"),
            ("/var/run/jambu", self.root / "run"),
            ("/opt/jambu", self.root / "opt"),
        ):
            rewritten = rewritten.replace(remote, str(local))
        script = self.root / "onstart.sh"
        script.write_text(rewritten)
        # On a real instance these paths are absolute; here the agent inherits
        # the rewritten locations through its documented env overrides.
        env = dict(os.environ)
        env.update(
            {
                "JAMBU_WATCHDOG_CONFIG": str(self.root / "etc" / "watchdog.json"),
                "JAMBU_WATCHDOG_STATE": str(self.root / "lib" / "activity.json"),
                "JAMBU_WATCHDOG_EVENTS": str(self.root / "log" / "events.jsonl"),
            }
        )
        subprocess.run(
            ["bash", str(script)],
            check=True,
            timeout=120,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def kill_watchdog(self) -> None:
        pid_file = self.root / "run" / "watchdog.pid"
        if not pid_file.exists():
            return
        try:
            os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
        except (OSError, ValueError):
            pass


@pytest.fixture
def booted(write_config, tmp_path, model_server):
    server, model_port = model_server
    watchdog_port = free_port()
    config = write_config(
        runtime={"engine": "vllm", "port": 8000, "context_length": 2048},
        lifecycle={
            "idle_timeout": "6s",
            "max_lifetime": "1h",
            "auto_stop": True,
            "heartbeat_interval": "1s",
            "heartbeat_grace": "5s",
            "watchdog": {"enabled": True, "port": watchdog_port, "interval": "1s"},
        },
        health={"interval": "2s", "startup_timeout": "60s"},
    )
    root = tmp_path / "instance"
    root.mkdir()
    marker = tmp_path / "marker.json"
    provider = BootingProvider(
        config, CredentialResolver(config.project_dir), root, marker, model_port
    )
    engine = Engine(config, provider=provider)
    yield engine, provider, server, marker
    provider.kill_watchdog()


def test_setup_boots_the_watchdog_waits_for_health_and_reports_ready(booted):
    engine, provider, server, marker = booted
    server.healthy = True

    state = engine.setup()

    assert state.state == InstanceState.READY.value
    assert state.endpoint and state.endpoint.startswith("http://127.0.0.1:")
    assert state.watchdog.installed and state.watchdog.url

    # The watchdog really is running, really knows who it is, and agrees we are ready.
    remote = engine.beacon(state).state()
    assert remote["instance_id"] == state.instance_id
    assert remote["starting"] is False
    assert remote["state"] in ("ready", "idle")

    events = [record["event"] for record in engine.events.tail(limit=50)]
    assert "instance_created" in events
    assert "runtime_ready" in events


def test_a_workload_reaches_the_remote_watchdog_and_holds_the_instance(booted):
    engine, provider, server, marker = booted
    server.healthy = True
    state = engine.setup()
    client = engine.beacon(state)

    # A workload that outlives the idle timeout must not be cut off.
    engine.run(
        [sys.executable, "-c", "import time; time.sleep(9)"], label="integration"
    )

    assert not marker.exists(), "the guard stopped an actively heartbeating workload"

    remote = client.state()
    workloads = remote["workloads"]
    assert len(workloads) == 1
    record = next(iter(workloads.values()))
    assert record["state"] == "finished"
    assert record["last_heartbeat"] > record["started_at"], "heartbeats did arrive"


def test_the_instance_stops_itself_once_idle(booted):
    engine, provider, server, marker = booted
    server.healthy = True
    engine.setup()

    deadline = time.time() + 40
    while time.time() < deadline and not marker.exists():
        time.sleep(0.5)

    assert marker.exists(), "the remote guard never stopped the idle instance"
    assert json.loads(marker.read_text())["action"] == "stop"

    # And the reason is on the record, on the instance itself.
    remote_events = (provider.root / "log" / "events.jsonl").read_text().splitlines()
    guard = [
        json.loads(line)
        for line in remote_events
        if line.strip() and json.loads(line)["event"] == "lifecycle_guard_triggered"
    ]
    assert guard and guard[-1]["reason"] == "idle_timeout"


def test_a_lock_survives_into_the_remote_guard(booted):
    engine, provider, server, marker = booted
    server.healthy = True
    state = engine.setup()

    engine.lock(30, reason="integration lock")
    assert engine.beacon(state).state()["locked"] is True

    time.sleep(12)  # twice the idle timeout
    assert not marker.exists(), "a locked instance must not be stopped"

    engine.unlock()
    deadline = time.time() + 30
    while time.time() < deadline and not marker.exists():
        time.sleep(0.5)
    assert marker.exists(), "unlocking must hand control back to the idle rule"


def test_setup_is_idempotent_against_a_live_instance(booted):
    engine, provider, server, marker = booted
    server.healthy = True

    first = engine.setup()
    second = engine.setup()
    assert first.instance_id == second.instance_id
    assert provider.counter == 1


def test_a_runtime_that_never_becomes_healthy_is_cleaned_up(booted):
    engine, provider, server, marker = booted
    server.healthy = False  # the model server never answers
    engine.config.health.startup_timeout = "8s"

    from gpu4agents.core.errors import RuntimeStartupError

    with pytest.raises(RuntimeStartupError):
        engine.setup()

    assert engine.store.load().instance_id is None, "partial resources must be cleaned up"
    events = [record["event"] for record in engine.events.tail(limit=50)]
    assert "setup_failed" in events
