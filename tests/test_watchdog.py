"""The remote watchdog, exercised for real: the generated script is executed
as its own process and driven over HTTP, exactly as it runs on the instance.

This is the guarantee behind spec criteria 6, 8, 9 and 12: lifecycle
enforcement does not depend on the local CLI staying alive.
"""

from __future__ import annotations

import http.client
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

from gpu4agents.core.beacon import WatchdogClient
from gpu4agents.core.policy import PolicyConfig
from gpu4agents.watchdog import agent_body
from gpu4agents.watchdog.builder import (
    build_agent_config,
    build_agent_source,
    build_onstart_script,
    generate_token,
)

# Provider actions that record what they were asked to do instead of calling an API.
MARKER_ACTIONS = '''
import json as _json
import os as _os


def _record(action, ctx):
    path = _os.environ["JAMBU_TEST_MARKER"]
    with open(path, "w") as handle:
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


# Provider actions that fail with a simulated HTTP 503 the first
# JAMBU_TEST_FAIL_<ACTION> times, logging every attempt.
FLAKY_ACTIONS = '''
import json as _json
import os as _os


def _attempt(action, ctx):
    path = _os.environ["JAMBU_TEST_ATTEMPTS"]
    with open(path, "a") as handle:
        handle.write(_json.dumps({"action": action}) + "\\n")
    with open(path) as handle:
        count = sum(1 for line in handle if _json.loads(line)["action"] == action)
    if count <= int(_os.environ.get("JAMBU_TEST_FAIL_" + action.upper(), "0")):
        raise RuntimeError("HTTP 503 Service Unavailable (simulated)")
    with open(_os.environ["JAMBU_TEST_MARKER"], "w") as handle:
        _json.dump({"action": action, "instance_id": ctx.get("instance_id")}, handle)


def provider_stop(ctx):
    _attempt("stop", ctx)


def provider_destroy(ctx):
    _attempt("destroy", ctx)
'''


class Agent:
    def __init__(
        self,
        tmp_path: Path,
        policy: PolicyConfig,
        *,
        actions: str = MARKER_ACTIONS,
        env: dict | None = None,
        state: dict | None = None,
        **overrides,
    ):
        self.dir = tmp_path
        self.token = generate_token()
        self.port = free_port()
        self.marker = tmp_path / "marker.json"
        self.events = tmp_path / "events.jsonl"
        self.attempts = tmp_path / "attempts.jsonl"
        self.state_path = tmp_path / "activity.json"

        config = build_agent_config(
            provider="test",
            instance_id="42",
            token=self.token,
            port=self.port,
            interval_s=overrides.pop("interval_s", 1.0),
            startup_grace_s=overrides.pop("startup_grace_s", 1.0),
            policy=policy,
            created_at=overrides.pop("created_at", time.time()),
            activity_probe=overrides.pop("activity_probe", None),
        )
        config.update(overrides)
        (tmp_path / "watchdog.json").write_text(json.dumps(config))
        (tmp_path / "agent.py").write_text(build_agent_source(actions))
        if state is not None:
            self.state_path.write_text(json.dumps(state))

        self.env = dict(os.environ)
        self.env.pop("JAMBU_SESSION_STARTED_AT", None)
        self.env.update(
            {
                "JAMBU_WATCHDOG_CONFIG": str(tmp_path / "watchdog.json"),
                "JAMBU_WATCHDOG_STATE": str(self.state_path),
                "JAMBU_WATCHDOG_EVENTS": str(self.events),
                "JAMBU_TEST_MARKER": str(self.marker),
                "JAMBU_TEST_ATTEMPTS": str(self.attempts),
            }
        )
        self.env.update(env or {})
        self.client = WatchdogClient(f"http://127.0.0.1:{self.port}", self.token, timeout=5.0)
        self.start()

    def start(self, env: dict | None = None) -> None:
        """(Re)launch the agent process against the same config and state."""
        self.env.update(env or {})
        self.log = self.dir / "agent.log"
        # A file, not a pipe: a long-running agent would block on a full pipe.
        with open(self.log, "a") as log:
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(self.dir / "agent.py")],
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self._await_boot()

    def _await_boot(self) -> None:
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"watchdog died: {self.log.read_text()}")
            if self.client.ping():
                return
            time.sleep(0.2)
        raise AssertionError("watchdog never came up")

    def records(self, event: str | None = None) -> list[dict]:
        if not self.events.exists():
            return []
        out = [json.loads(line) for line in self.events.read_text().splitlines() if line.strip()]
        return [r for r in out if event is None or r["event"] == event]

    def attempted(self) -> list[str]:
        if not self.attempts.exists():
            return []
        return [json.loads(line)["action"] for line in self.attempts.read_text().splitlines()]

    def wait_for_marker(self, timeout: float = 20.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.marker.exists():
                return json.loads(self.marker.read_text())
            time.sleep(0.2)
        return None

    def event_names(self) -> list[str]:
        if not self.events.exists():
            return []
        return [
            json.loads(line)["event"]
            for line in self.events.read_text().splitlines()
            if line.strip()
        ]

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.process.kill()


@pytest.fixture
def agent_factory(tmp_path):
    agents: list[Agent] = []

    def _make(policy: PolicyConfig, **overrides) -> Agent:
        directory = tmp_path / f"agent{len(agents)}"
        directory.mkdir()
        agent = Agent(directory, policy, **overrides)
        agents.append(agent)
        return agent

    yield _make
    for agent in agents:
        agent.close()


def policy(**overrides) -> PolicyConfig:
    base = dict(idle_timeout_s=2.0, max_lifetime_s=None, heartbeat_grace_s=2.0)
    base.update(overrides)
    return PolicyConfig(**base)


# ---------------------------------------------------------------------------


def test_generated_agent_is_valid_python():
    source = build_agent_source(MARKER_ACTIONS)
    compile(source, "watchdog.py", "exec")
    assert "def evaluate(" in source  # the core policy really is inlined
    assert "def provider_stop(" in source


def test_the_control_port_requires_the_token(agent_factory):
    agent = agent_factory(policy(idle_timeout_s=3600.0))
    rogue = WatchdogClient(agent.client.base_url, "wrong-token", timeout=5.0)
    assert rogue.ping() is True          # /health is open, for liveness probes
    assert rogue.try_state() is None     # everything else is not


def test_idle_instance_is_stopped_with_a_recorded_reason(agent_factory):
    agent = agent_factory(policy())
    agent.client.mark_ready()

    result = agent.wait_for_marker()
    assert result == {"action": "stop", "instance_id": "42"}

    events = agent.event_names()
    assert "lifecycle_guard_triggered" in events
    assert "instance_stopped" in events

    records = [
        json.loads(line)
        for line in agent.events.read_text().splitlines()
        if line.strip()
    ]
    guard = [r for r in records if r["event"] == "lifecycle_guard_triggered"][-1]
    assert guard["reason"] == "idle_timeout"
    assert guard["action"] == "stop"


def test_a_heartbeating_workload_is_never_stopped(agent_factory):
    agent = agent_factory(policy())
    agent.client.mark_ready()
    agent.client.workload_start("w1", "long eval")

    # Heartbeat across several idle windows.
    for _ in range(12):
        agent.client.workload_heartbeat("w1")
        time.sleep(0.5)

    assert not agent.marker.exists()
    assert agent.client.state()["state"] == "busy"

    # Finish it and the instance goes idle, then stops.
    agent.client.workload_end("w1", 0)
    assert agent.wait_for_marker()["action"] == "stop"


def test_a_vanished_heartbeat_falls_back_to_idle(agent_factory):
    """Criterion 9: the local CLI crashed mid-workload."""
    agent = agent_factory(policy(heartbeat_grace_s=2.0))
    agent.client.mark_ready()
    agent.client.workload_start("w1", "crashed run")
    # No further heartbeats, no workload_end: exactly what a crash looks like.
    assert agent.wait_for_marker(timeout=25)["action"] == "stop"


def test_a_lock_prevents_shutdown_until_it_expires(agent_factory):
    agent = agent_factory(policy())
    agent.client.mark_ready()
    agent.client.lock(ttl_seconds=6, reason="benchmark")

    time.sleep(4)
    assert not agent.marker.exists(), "locked instance must not be stopped"
    assert agent.client.state()["locked"] is True

    assert agent.wait_for_marker(timeout=25)["action"] == "stop"
    assert "instance_unlocked" in agent.event_names()


def test_indefinite_locks_are_refused_by_default(agent_factory):
    agent = agent_factory(policy(idle_timeout_s=3600.0))
    assert agent.client.try_request("POST", "/lock", {"ttl_seconds": None}) is None


def test_indefinite_locks_work_when_explicitly_allowed(agent_factory):
    agent = agent_factory(policy(idle_timeout_s=1.0, allow_indefinite_lock=True))
    agent.client.mark_ready()
    agent.client.lock(ttl_seconds=None, reason="pinned")
    time.sleep(4)
    assert not agent.marker.exists()


def test_max_lifetime_stops_a_busy_instance(agent_factory):
    agent = agent_factory(
        policy(idle_timeout_s=3600.0, max_lifetime_s=3.0),
        created_at=time.time(),
    )
    agent.client.mark_ready()
    agent.client.workload_start("w1", "endless")

    deadline = time.time() + 20
    while time.time() < deadline and not agent.marker.exists():
        agent.client.workload_heartbeat("w1")
        time.sleep(0.5)

    result = agent.wait_for_marker(timeout=5)
    assert result is not None and result["action"] == "stop"


def test_auto_destroy_escalates(agent_factory):
    agent = agent_factory(policy(auto_destroy=True))
    agent.client.mark_ready()
    assert agent.wait_for_marker()["action"] == "destroy"


def test_startup_grace_expires_when_the_cli_never_confirms(agent_factory):
    """The CLI died during setup: the instance must not stay 'starting' forever."""
    agent = agent_factory(policy(), startup_grace_s=2.0)
    # Never call /ready.
    assert agent.wait_for_marker(timeout=25)["action"] == "stop"
    assert "runtime_startup_grace_expired" in agent.event_names()


def test_identify_teaches_the_watchdog_who_it_is(agent_factory):
    agent = agent_factory(policy(idle_timeout_s=3600.0))
    agent.client.identify("99", hourly_cost_usd=1.25)
    state = agent.client.state()
    assert state["instance_id"] == "99"
    assert state["policy"]["hourly_cost_usd"] == 1.25


def test_manual_shutdown_route(agent_factory):
    agent = agent_factory(policy(idle_timeout_s=3600.0))
    agent.client.shutdown("destroy", reason="manual")
    assert agent.wait_for_marker()["action"] == "destroy"


def test_onstart_script_starts_the_watchdog_before_the_model(agent_factory):
    script = build_onstart_script(
        agent_source="print('agent')",
        agent_config={"port": 8777},
        runtime_start_command="nohup vllm serve &",
        runtime_env={"HF_HOME": "/workspace/hf"},
    )
    assert script.startswith("#!/bin/bash")
    assert script.index("watchdog.py") < script.index("vllm serve")
    assert "chmod 600 /etc/jambu/watchdog.json" in script
    assert "export HF_HOME='/workspace/hf'" in script


def test_the_boot_script_reconstitutes_the_agent_byte_for_byte(tmp_path):
    """The agent travels compressed inside a provider API field; prove it survives."""
    source = build_agent_source(MARKER_ACTIONS)
    script = build_onstart_script(
        agent_source=source,
        agent_config={"token": "t", "port": 8777},
        runtime_start_command="echo model",
    )
    assert len(script) < len(source), "the payload must be compressed, not inlined raw"

    root = tmp_path / "root"
    boot = write_local_boot_script(tmp_path, script, root)
    try:
        subprocess.run(["bash", str(boot)], check=True, timeout=60)
        assert (root / "opt" / "watchdog.py").read_text() == source
        # The config (it carries the token) travels deflated too, readable by root only.
        written = root / "etc" / "watchdog.json"
        assert json.loads(written.read_text()) == {"token": "t", "port": 8777}
        assert written.stat().st_mode & 0o777 == 0o600
        assert not list((root / "etc").glob("*.b85"))
    finally:
        kill_supervisor(root)


def write_local_boot_script(tmp_path: Path, script: str, root: Path) -> Path:
    """The onstart script with its absolute paths moved under ``root``."""
    rewritten = script
    for remote, local in (
        ("/etc/jambu", root / "etc"),
        ("/var/lib/jambu", root / "lib"),
        ("/var/log/jambu", root / "log"),
        ("/var/run/jambu", root / "run"),
        ("/opt/jambu", root / "opt"),
    ):
        rewritten = rewritten.replace(remote, str(local))
    boot = tmp_path / "onstart.sh"
    boot.write_text(rewritten)
    return boot


def _children(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text()
    except OSError:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
        text = out.stdout
    return [int(p) for p in text.split()]


def kill_supervisor(root: Path) -> None:
    pid_file = root / "run" / "watchdog.pid"
    if not pid_file.exists():
        return
    pid = int(pid_file.read_text().strip())
    children = _children(pid)
    for target in [pid] + children:
        try:
            os.kill(target, signal.SIGKILL)
        except OSError:
            pass


# --- restart sessions --------------------------------------------------------


def persisted_previous_session(now: float, session: float, **extra) -> dict:
    """activity.json as a stopped instance leaves it on its container disk."""
    state = {
        "created_at": now - 10_000,
        "session_started_at": session,
        "last_activity": now - 9_000,
        "starting": False,
        "locked": False,
        "locked_until": None,
        "terminated": True,
        "terminated_at": now - 9_000,
        "workloads": {
            "w-old": {
                "started_at": now - 9_500,
                "last_heartbeat": now - 9_100,
                "label": "interrupted by the stop",
                "state": "running",
            }
        },
    }
    state.update(extra)
    return state


def test_a_restarted_instance_is_guarded_again(agent_factory):
    """Bug: terminated=True survived the stop/start and disarmed the guard."""
    now = time.time()
    session = str(int(now))
    agent = agent_factory(
        policy(max_lifetime_s=3_600.0),
        created_at=now - 10_000,
        state=persisted_previous_session(now, session=now - 10_000),
        env={"JAMBU_SESSION_STARTED_AT": session},
        startup_grace_s=600.0,
    )
    state = agent.client.state()
    assert state["terminated"] is False
    assert state["session_started_at"] == float(session)
    assert state["session_age_seconds"] < 60
    assert state["age_seconds"] > 9_000  # creation age is still the truth
    assert state["starting"] is True
    assert state["workloads"]["w-old"]["state"] == "failed"
    assert "session_started" in agent.event_names()

    agent.client.mark_ready()
    assert agent.wait_for_marker()["action"] == "stop"
    # Measured from the session, max_lifetime must not have been the trigger.
    assert agent.records("lifecycle_guard_triggered")[-1]["reason"] == "idle_timeout"


def test_a_supervisor_respawn_in_the_same_session_keeps_state(agent_factory):
    now = time.time()
    session = int(now) - 30
    agent = agent_factory(
        policy(),
        state=persisted_previous_session(now, session=float(session), terminated_at=now),
        env={"JAMBU_SESSION_STARTED_AT": str(session)},
    )
    time.sleep(3)
    state = agent.client.state()
    assert state["terminated"] is True
    assert state["session_started_at"] == float(session)
    assert state["workloads"]["w-old"]["state"] == "running"
    assert not agent.marker.exists(), "a respawn must not re-fire a fresh stop"
    assert "session_started" not in agent.event_names()


def test_a_crashed_agent_respawned_by_hand_keeps_its_session(agent_factory):
    session = str(int(time.time()))
    agent = agent_factory(
        policy(idle_timeout_s=3_600.0), env={"JAMBU_SESSION_STARTED_AT": session}
    )
    agent.client.mark_ready()
    agent.client.lock(ttl_seconds=600, reason="keep")
    agent.process.kill()
    agent.process.wait()
    agent.start()
    state = agent.client.state()
    assert state["locked"] is True
    assert state["starting"] is False
    assert agent.event_names().count("session_started") == 1


# --- retries and escalation --------------------------------------------------


def test_the_guard_survives_a_failed_stop_and_retries(agent_factory):
    """Bug: one transient 503 used to end the guard loop for good."""
    agent = agent_factory(
        policy(), actions=FLAKY_ACTIONS, env={"JAMBU_TEST_FAIL_STOP": "1"}
    )
    agent.client.mark_ready()
    assert agent.wait_for_marker(timeout=25)["action"] == "stop"
    assert agent.attempted() == ["stop", "stop"]
    names = agent.event_names()
    assert "provider_error" in names
    retry = agent.records("guard_retry")[0]
    assert retry["attempt"] == 1 and retry["action"] == "stop"
    assert agent.client.state()["terminated"] is True


def test_stops_failing_past_escalate_after_s_escalate_to_destroy(agent_factory):
    agent = agent_factory(
        policy(),
        actions=FLAKY_ACTIONS,
        env={"JAMBU_TEST_FAIL_STOP": "999"},
        escalate_after_s=2.5,
    )
    agent.client.mark_ready()
    assert agent.wait_for_marker(timeout=30)["action"] == "destroy"
    attempts = agent.attempted()
    assert attempts[-1] == "destroy" and set(attempts[:-1]) == {"stop"}
    assert len(attempts) >= 3
    trigger = agent.records("lifecycle_guard_triggered")[-1]
    assert trigger["reason"] == "stop_failed_escalation"
    assert trigger["original_reason"] == "idle_timeout"


# In-process: the escalation clock, without waiting real minutes.


@pytest.fixture
def inproc(tmp_path, monkeypatch):
    calls = []
    failing = {"stop": True}

    def stop(ctx):
        calls.append("stop")
        if failing["stop"]:
            raise RuntimeError("vast 503")

    def destroy(ctx):
        calls.append("destroy")

    monkeypatch.setattr(agent_body, "STATE_PATH", str(tmp_path / "activity.json"))
    monkeypatch.setattr(agent_body, "EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(agent_body, "provider_stop", stop, raising=False)
    monkeypatch.setattr(agent_body, "provider_destroy", destroy, raising=False)
    monkeypatch.setattr(
        agent_body,
        "_CONFIG",
        {"policy": {"idle_timeout_s": 60, "auto_stop": True, "auto_destroy": False}, "interval_s": 60},
    )
    monkeypatch.setattr(agent_body, "_STATE", {})
    agent_body._STATE.update(agent_body.load_state())
    agent_body._STATE["starting"] = False

    def idle(seconds=3600.0):
        agent_body._STATE["last_activity"] = time.time() - seconds

    def tick(**_):
        agent_body.guard_tick(60, False)

    return agent_body, calls, failing, idle, tick


def test_an_api_blip_does_not_destroy_the_disk(inproc):
    """Bug: three failed stops (three minutes of a Vast API outage) destroyed the disk."""
    agent, calls, _failing, idle, tick = inproc
    idle()
    for _ in range(5):
        tick()
    assert calls == ["stop"] * 5
    assert agent._STATE["stop_failing_since"] is not None


def test_failures_reset_when_the_instance_is_back_in_use(inproc):
    """Bug: stop_failures never reset, so failures from separate idle spells,
    hours apart, added up to an escalation."""
    agent, calls, _failing, idle, tick = inproc
    idle()
    tick()
    tick()
    agent._STATE["last_activity"] = time.time()  # the user comes back
    tick()
    assert agent._STATE["stop_failures"] == 0 and agent._STATE["stop_failing_since"] is None
    idle()
    agent._CONFIG["escalate_after_s"] = 10.0
    tick()
    assert calls == ["stop", "stop", "stop"]


def test_stops_failing_continuously_for_escalate_after_s_destroy(inproc):
    agent, calls, _failing, idle, tick = inproc
    idle()
    tick()
    agent._STATE["stop_failing_since"] = time.time() - 1801  # half an hour of 503s
    tick()
    assert calls == ["stop", "destroy"]
    assert agent._STATE["terminated_reason"] == "stop_failed_escalation"


def test_no_escalation_without_destroy_support(inproc):
    agent, calls, _failing, idle, tick = inproc
    agent._CONFIG["policy"]["destroy_supported"] = False
    idle()
    tick()
    agent._STATE["stop_failing_since"] = time.time() - 7200
    tick()
    assert calls == ["stop", "stop"]


def test_a_successful_terminate_clears_the_pending_shutdown(inproc):
    agent, calls, failing, _idle, tick = inproc
    agent._STATE["pending_shutdown"] = {"action": "stop", "reason": "manual"}
    tick()
    assert agent._STATE["pending_shutdown"] is not None  # failed: still to retry
    failing["stop"] = False
    agent._STATE["terminated"] = False
    tick()
    assert calls == ["stop", "stop"]
    assert agent._STATE["terminated"] is True
    assert agent._STATE["pending_shutdown"] is None
    assert agent._STATE["stop_failures"] == 0


def test_no_escalation_when_destroy_is_unsupported(agent_factory):
    agent = agent_factory(
        policy(destroy_supported=False),
        actions=FLAKY_ACTIONS,
        env={"JAMBU_TEST_FAIL_STOP": "999"},
    )
    agent.client.mark_ready()
    deadline = time.time() + 25
    while time.time() < deadline and len(agent.attempted()) < 5:
        time.sleep(0.3)
    assert set(agent.attempted()) == {"stop"}
    assert len(agent.attempted()) >= 5, "the guard must keep retrying"


def test_an_accepted_stop_is_reissued_when_the_machine_stays_up(agent_factory):
    agent = agent_factory(policy(), actions=FLAKY_ACTIONS, reissue_after_s=2.0)
    agent.client.mark_ready()
    assert agent.wait_for_marker()["action"] == "stop"
    deadline = time.time() + 20
    while time.time() < deadline and len(agent.attempted()) < 2:
        time.sleep(0.3)
    assert agent.attempted()[:2] == ["stop", "stop"]
    assert "guard_reissue" in agent.event_names()


def test_a_failed_manual_shutdown_is_retried_by_the_guard(agent_factory):
    agent = agent_factory(
        policy(idle_timeout_s=3_600.0),
        actions=FLAKY_ACTIONS,
        env={"JAMBU_TEST_FAIL_DESTROY": "1"},
    )
    agent.client.shutdown("destroy", reason="manual")
    assert agent.wait_for_marker(timeout=25)["action"] == "destroy"
    assert agent.attempted() == ["destroy", "destroy"]


# --- supervisor --------------------------------------------------------------


def test_onstart_wraps_the_watchdog_in_a_supervisor():
    script = build_onstart_script(
        agent_source="print('agent')",
        agent_config={"port": 8777},
        runtime_start_command="nohup vllm serve &",
    )
    subprocess.run(["bash", "-n", "-c", script], check=True)
    assert "export JAMBU_SESSION_STARTED_AT=$(date +%s)" in script
    assert "while true; do" in script
    assert "sleep 5" in script
    assert "echo $! > /var/run/jambu/watchdog.pid" in script
    assert script.index("JAMBU_SESSION_STARTED_AT") < script.index("while true")
    assert script.index("while true") < script.index("vllm serve")


def _ping(port: int) -> bool:
    return WatchdogClient(f"http://127.0.0.1:{port}", "x", timeout=1.0).ping()


def test_the_supervisor_respawns_a_killed_watchdog(tmp_path):
    port = free_port()
    root = tmp_path / "root"
    config = build_agent_config(
        provider="test",
        instance_id="42",
        token=generate_token(),
        port=port,
        interval_s=1.0,
        startup_grace_s=600.0,
        policy=policy(idle_timeout_s=3_600.0),
        created_at=time.time(),
    )
    script = build_onstart_script(
        agent_source=build_agent_source(MARKER_ACTIONS),
        agent_config=config,
        runtime_start_command="echo model",
    )
    boot = write_local_boot_script(tmp_path, script, root)
    env = dict(os.environ)
    env.pop("JAMBU_SESSION_STARTED_AT", None)
    env.update(
        {
            "JAMBU_WATCHDOG_CONFIG": str(root / "etc" / "watchdog.json"),
            "JAMBU_WATCHDOG_STATE": str(root / "lib" / "activity.json"),
            "JAMBU_WATCHDOG_EVENTS": str(root / "log" / "events.jsonl"),
            "JAMBU_TEST_MARKER": str(tmp_path / "marker.json"),
        }
    )
    events = root / "log" / "events.jsonl"

    def wait(predicate, timeout=20.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.2)
        return False

    try:
        subprocess.run(["bash", str(boot)], check=True, timeout=60, env=env)
        assert wait(lambda: _ping(port)), "watchdog never came up under the supervisor"
        supervisor = int((root / "run" / "watchdog.pid").read_text().strip())
        (first,) = _children(supervisor)
        os.kill(first, signal.SIGKILL)
        assert wait(lambda: _children(supervisor) and _children(supervisor) != [first])
        assert wait(lambda: _ping(port)), "the supervisor did not respawn the watchdog"

        names = [json.loads(line)["event"] for line in events.read_text().splitlines()]
        assert names.count("watchdog_started") == 2
        # Same container start, same session: the respawn must not reset state.
        assert names.count("session_started") == 1
        assert "respawning" in (root / "log" / "watchdog.log").read_text()

        # `kill $(cat watchdog.pid)` still means "stop the watchdog".
        (second,) = _children(supervisor)
        os.kill(supervisor, signal.SIGTERM)
        assert wait(lambda: not Path(f"/proc/{second}").exists() and not _ping(port))
    finally:
        kill_supervisor(root)


# --- inference traffic as activity -------------------------------------------


VLLM_METRICS = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model_name="Qwen/Qwen2.5-7B"}} {running}
vllm:num_requests_waiting{{model_name="Qwen/Qwen2.5-7B"}} 0.0
# TYPE vllm:request_success_total counter
vllm:request_success_total{{finished_reason="stop",model_name="Qwen/Qwen2.5-7B"}} {ok}
vllm:request_success_total{{finished_reason="length",model_name="Qwen/Qwen2.5-7B"}} 1.0
vllm:prompt_tokens_total{{model_name="Qwen/Qwen2.5-7B"}} {tokens} 1700000000000
"""


class FakeMetrics:
    def __init__(self):
        self.running = 0
        self.ok = 0
        self.tokens = 0
        metrics = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_GET(self):  # noqa: N802
                body = VLLM_METRICS.format(
                    running=float(metrics.running), ok=float(metrics.ok), tokens=float(metrics.tokens)
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/metrics"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def probe(self) -> dict:
        return {
            "kind": "prometheus",
            "url": self.url,
            "gauges": ["vllm:num_requests_running", "vllm:num_requests_waiting"],
            "counters": ["vllm:request_success_total", "vllm:prompt_tokens_total"],
        }

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_metrics():
    metrics = FakeMetrics()
    yield metrics
    metrics.close()


def test_direct_inference_traffic_keeps_the_instance_alive(agent_factory, fake_metrics):
    """An agent calling vLLM directly, never through `gpu run`, is still use."""
    agent = agent_factory(policy(idle_timeout_s=3.0), activity_probe=fake_metrics.probe())
    agent.client.mark_ready()

    for _ in range(16):  # ~8s of traffic, well past the 3s idle timeout
        fake_metrics.ok += 1
        fake_metrics.tokens += 50
        time.sleep(0.5)
    assert not agent.marker.exists(), "served traffic must count as activity"

    probe = agent.client.state()["last_probe"]
    assert probe["ok"] is True
    assert probe["last_traffic_at"] is not None
    assert probe["values"]["vllm:request_success_total"] > 1.0

    # Traffic ends: the normal idle rule takes over.
    assert agent.wait_for_marker(timeout=20)["action"] == "stop"
    assert agent.records("lifecycle_guard_triggered")[-1]["reason"] == "idle_timeout"


def test_an_in_flight_request_keeps_the_instance_alive(agent_factory, fake_metrics):
    fake_metrics.running = 1  # one long generation, counters flat
    agent = agent_factory(policy(idle_timeout_s=2.0), activity_probe=fake_metrics.probe())
    agent.client.mark_ready()
    time.sleep(5)
    assert not agent.marker.exists()
    fake_metrics.running = 0
    assert agent.wait_for_marker(timeout=20)["action"] == "stop"


def test_a_flat_metrics_endpoint_is_not_activity(agent_factory, fake_metrics):
    fake_metrics.ok = 7  # pre-existing totals only establish the baseline
    agent = agent_factory(policy(), activity_probe=fake_metrics.probe())
    agent.client.mark_ready()
    assert agent.wait_for_marker()["action"] == "stop"
    assert agent.client.state()["last_probe"]["last_traffic_at"] is None


def test_probe_errors_are_not_activity_and_are_logged_once(agent_factory):
    dead = f"http://127.0.0.1:{free_port()}/metrics"
    agent = agent_factory(
        policy(), activity_probe={"kind": "prometheus", "url": dead, "gauges": [], "counters": []}
    )
    agent.client.mark_ready()
    assert agent.wait_for_marker()["action"] == "stop"
    time.sleep(1.5)
    assert agent.event_names().count("activity_probe_error") == 1
    assert agent.client.state()["last_probe"]["ok"] is False


def test_watchdog_health_hits_are_not_activity(agent_factory):
    agent = agent_factory(policy())
    agent.client.mark_ready()
    deadline = time.time() + 20
    while time.time() < deadline and not agent.marker.exists():
        agent.client.ping()
        time.sleep(0.2)
    assert agent.marker.exists()


def test_build_agent_config_carries_the_activity_probe():
    probe = {"kind": "prometheus", "url": "http://127.0.0.1:8000/metrics", "gauges": [], "counters": []}
    kwargs = dict(
        provider="test",
        instance_id="1",
        token="t",
        port=1,
        interval_s=1.0,
        startup_grace_s=1.0,
        policy=policy(),
        created_at=0.0,
    )
    assert "activity_probe" not in build_agent_config(**kwargs)
    assert build_agent_config(activity_probe=probe, **kwargs)["activity_probe"] == probe


# --- prometheus parsing (pure functions, imported directly) ------------------


def test_prometheus_samples_are_summed_across_label_sets():
    totals = agent_body.parse_prometheus(VLLM_METRICS.format(running=2.0, ok=3.0, tokens=40.0))
    assert totals["vllm:num_requests_running"] == 2.0
    assert totals["vllm:request_success_total"] == 4.0
    assert totals["vllm:prompt_tokens_total"] == 40.0  # trailing timestamp ignored


def test_prometheus_parser_tolerates_awkward_lines():
    text = "\n".join(
        [
            "# HELP x something {with braces}",
            "",
            'weird{path="/a}b",note="say \\"hi\\""} 5',
            "plain 1.5",
            "plain 2",
            "broken_no_value",
            "nan_metric NaN",
            "bad_value abc",
        ]
    )
    totals = agent_body.parse_prometheus(text)
    assert totals == {"weird": 5.0, "plain": 3.5}


def test_metric_lookup_tolerates_total_suffixes():
    totals = {"vllm:request_success_total": 3.0, "tokens": 9.0}
    assert agent_body.metric_value(totals, "vllm:request_success") == 3.0
    assert agent_body.metric_value(totals, "tokens_total") == 9.0
    assert agent_body.metric_value(totals, "missing") is None


def test_traffic_detection_rules():
    seen = agent_body.traffic_seen
    gauges, counters = ["running"], ["done_total"]
    assert seen({"running": 1.0}, None, gauges, counters) is True
    assert seen({"done_total": 5.0}, None, gauges, counters) is False  # baseline only
    assert seen({"done_total": 6.0}, {"done_total": 5.0}, gauges, counters) is True
    assert seen({"done_total": 5.0}, {"done_total": 5.0}, gauges, counters) is False
    assert seen({"done_total": 1.0}, {"done_total": 5.0}, gauges, counters) is False  # reset
    assert seen({}, {}, gauges, counters) is False


# --- token -------------------------------------------------------------------


def test_the_token_compare_rejects_near_misses(agent_factory):
    agent = agent_factory(policy(idle_timeout_s=3600.0))
    for bad in ("", agent.token[:-1], agent.token + "x", agent.token[:-1] + "!"):
        rogue = WatchdogClient(agent.client.base_url, bad, timeout=5.0)
        assert rogue.try_state() is None, bad
    # Non-ASCII junk must be a clean 401, not a crashed handler.
    conn = http.client.HTTPConnection("127.0.0.1", agent.port, timeout=5)
    conn.putrequest("GET", "/state")
    conn.putheader("X-Jambu-Token", "ünïcode".encode("latin-1"))
    conn.endheaders()
    assert conn.getresponse().status == 401
    conn.close()
    assert agent.client.try_state() is not None
    assert "hmac.compare_digest" in build_agent_source()


# --- runtime liveness and instance logs --------------------------------------


def _get(agent, path):
    conn = http.client.HTTPConnection("127.0.0.1", agent.port, timeout=5)
    conn.request("GET", path, headers={"X-Jambu-Token": agent.token})
    response = conn.getresponse()
    body = json.loads(response.read() or b"{}")
    conn.close()
    return response.status, body


def _wait(predicate, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


def test_logs_serves_the_tail_of_each_configured_log(agent_factory, tmp_path):
    runtime_log = tmp_path / "vllm.log"
    runtime_log.write_text("".join(f"line {i}\n" for i in range(5000)))
    onstart_log = tmp_path / "onstart.log"
    onstart_log.write_text("begin\ndone\n")
    agent = agent_factory(
        policy(idle_timeout_s=3600.0),
        logs={
            "runtime": str(runtime_log),
            "onstart": str(onstart_log),
            "watchdog": str(tmp_path / "missing.log"),
        },
    )

    assert agent.client.logs("runtime", 3) == ["line 4997", "line 4998", "line 4999"]
    status, body = _get(agent, "/logs?name=runtime")
    assert status == 200 and body["name"] == "runtime"
    assert len(body["lines"]) == 200 and body["lines"][-1] == "line 4999"
    _, capped = _get(agent, "/logs?name=runtime&lines=999999")
    assert len(capped["lines"]) == 2000 and capped["lines"][0] == "line 3000"
    assert agent.client.logs("onstart", 50) == ["begin", "done"]
    assert agent.client.logs("watchdog", 10) == []  # not written yet: empty, not an error
    assert _get(agent, "/logs?name=passwd")[0] == 404
    assert _get(agent, "/logs?name=runtime&lines=junk")[0] == 200

    rogue = WatchdogClient(agent.client.base_url, "wrong", timeout=5.0)
    with pytest.raises(Exception, match="token"):
        rogue.logs("runtime", 5)


def test_the_tail_is_read_from_the_end_of_a_large_file(tmp_path):
    big = tmp_path / "big.log"
    with open(big, "w") as handle:
        for i in range(200_000):
            handle.write(f"{i:08d} " + "x" * 80 + "\n")
    tail = agent_body.tail_lines(str(big), 3)
    assert [line.split()[0] for line in tail] == ["00199997", "00199998", "00199999"]
    assert agent_body.tail_lines(str(big), 0) == []
    assert agent_body.tail_lines(str(tmp_path / "nope.log"), 5) == []
    short = tmp_path / "short.log"
    short.write_text("only\n")
    assert agent_body.tail_lines(str(short), 10) == ["only"]


def test_runtime_alive_follows_the_model_server_process(agent_factory, tmp_path):
    pid_path = tmp_path / "vllm.pid"
    agent = agent_factory(
        policy(idle_timeout_s=3600.0),
        startup_grace_s=600.0,
        runtime_pid_path=str(pid_path),
    )
    state = agent.client.state()
    assert state["runtime_alive"] is None  # no pid file yet: unknown, not dead
    assert state["starting"] is True

    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        pid_path.write_text(f"{server.pid}\n")
        assert agent.client.state()["runtime_alive"] is True
        assert agent.client.state()["starting"] is True
    finally:
        server.kill()
        server.wait()

    # A dead model server must not hold the startup grace open.
    assert _wait(lambda: agent.client.state()["runtime_alive"] is False)
    assert _wait(lambda: agent.client.state()["starting"] is False)
    assert _wait(lambda: len(agent.records("runtime_exited")) == 1)
    time.sleep(2.5)  # a few guard ticks later, still logged only once
    assert len(agent.records("runtime_exited")) == 1
    assert json.loads(agent.state_path.read_text())["starting"] is False


def test_a_dead_model_server_lets_the_idle_rule_stop_the_machine(agent_factory, tmp_path):
    pid_path = tmp_path / "vllm.pid"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    pid_path.write_text(str(dead.pid))
    agent = agent_factory(
        policy(idle_timeout_s=2.0),
        startup_grace_s=1500.0,
        runtime_pid_path=str(pid_path),
    )
    assert agent.wait_for_marker()["action"] == "stop"
    assert "runtime_exited" in agent.event_names()
    assert "runtime_startup_grace_expired" not in agent.event_names()


def test_a_pid_file_from_a_previous_session_is_not_trusted(agent_factory, tmp_path):
    """Container disk survives a stop; the old pid file predates this start."""
    pid_path = tmp_path / "vllm.pid"
    pid_path.write_text("999999")
    old = time.time() - 3600
    os.utime(pid_path, (old, old))
    agent = agent_factory(
        policy(idle_timeout_s=3600.0),
        startup_grace_s=600.0,
        runtime_pid_path=str(pid_path),
    )
    time.sleep(1.5)
    state = agent.client.state()
    assert state["runtime_alive"] is None
    assert state["starting"] is True
    assert "runtime_exited" not in agent.event_names()


def test_without_a_pid_path_liveness_is_unknown(agent_factory):
    agent = agent_factory(policy(idle_timeout_s=3600.0), startup_grace_s=600.0)
    assert agent.client.state()["runtime_alive"] is None


def test_the_agent_config_names_the_runtime_files():
    config = build_agent_config(
        provider="test",
        instance_id="",
        token="t",
        port=8777,
        interval_s=60.0,
        startup_grace_s=900.0,
        policy=policy(),
        created_at=0.0,
        runtime_pid_path="/var/run/jambu/vllm.pid",
        runtime_log_path="/var/log/jambu/vllm.log",
    )
    assert config["runtime_pid_path"] == "/var/run/jambu/vllm.pid"
    assert config["logs"] == {
        "runtime": "/var/log/jambu/vllm.log",
        "onstart": "/var/log/jambu/onstart.log",
        "watchdog": "/var/log/jambu/watchdog.log",
    }
    bare = build_agent_config(
        provider="test", instance_id="", token="t", port=8777, interval_s=60.0,
        startup_grace_s=900.0, policy=policy(), created_at=0.0,
    )
    assert "runtime_pid_path" not in bare and "runtime" not in bare["logs"]
