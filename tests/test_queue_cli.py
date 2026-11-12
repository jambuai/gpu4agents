"""`gpu queue ...`, the Python API and the controls, on a real LocalBackend and output store
with the pool replaced by FakePool (tests/queue_fakes.py)."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpu4agents.agent.mcp import Server
from gpu4agents.cli import _common, timer
from gpu4agents.cli.main import app
from gpu4agents.core import job as jobmod
from gpu4agents.core.events import EventLog
from gpu4agents.queue import api
from gpu4agents.queue.backends.local import LocalBackend
from gpu4agents.queue.coordinator import Coordinator
from gpu4agents.queue.models import JobState, QueueJob
from gpu4agents.queue.observe import Observer
from gpu4agents.queue.outputs import LocalOutputStore

from queue_fakes import FakePool, FakeRun

runner = CliRunner()


class Clock:
    def __init__(self) -> None:
        self.t = time.time()

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class Env:
    def __init__(self, tmp_path: Path, cfg, monkeypatch) -> None:
        self.tmp = tmp_path
        self.cfg = cfg
        self.path = tmp_path / "gpu.yaml"
        self.clock = Clock()
        self.backend = LocalBackend(cfg.state_dir / "queue" / "queue.db", clock=self.clock)
        self.outputs = LocalOutputStore(cfg.project_dir / "gpu-outputs", bundles_dir=cfg.state_dir / "queue" / "bundles")
        self.obs = Observer(EventLog(cfg.state_dir / "logs"), cfg.state_dir / "queue" / "metrics.prom")
        self.pool = FakePool(self.obs, self.clock, tmp_path / "machines")
        self.pool.advance = self.clock.advance
        monkeypatch.setattr(api, "build_pool", lambda config, observer: self.pool)
        monkeypatch.setattr(api, "load_backend", lambda spec, config: self.backend)
        monkeypatch.setenv("GPU_QUEUE_SUBMITTER", "alice")
        monkeypatch.setenv("GPU_NO_STOPPED_CHECK", "1")

    def job_dir(self, name: str = "job", run: str | None = "bash run.sh", **manifest) -> Path:
        d = self.tmp / name
        d.mkdir(exist_ok=True)
        (d / "run.sh").write_text("echo hi\n")
        lines = ([f"run: {run}"] if run else []) + [f"{k}: {v}" for k, v in manifest.items()]
        (d / "gpujob.yaml").write_text("\n".join(lines) + "\n")
        return d

    def cli(self, *args: str, expect: int | None = 0):
        result = runner.invoke(app, ["--json", "--config", str(self.path), *args])
        if expect is not None:
            assert result.exit_code == expect, result.output
        return result

    def data(self, *args: str):
        return json.loads(self.cli(*args).stdout)

    def coordinator(self, worker: str = "w1") -> Coordinator:
        return Coordinator(self.cfg, self.backend, self.outputs, self.pool, self.obs, worker_id=worker, clock=self.clock)

    def step(self, coord: Coordinator, n: int = 1) -> None:
        for _ in range(n):
            self.clock.advance(5)
            coord.tick()

    def until(self, coord: Coordinator, pred, n: int = 40) -> None:
        for _ in range(n):
            if pred():
                return
            self.step(coord)
        assert pred()

    def events(self, name: str) -> list[dict]:
        path = self.cfg.state_dir / "logs" / "events.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        return [r for r in rows if r["event"] == name]


@pytest.fixture
def env(write_config, tmp_path, monkeypatch):
    cfg = write_config(runtime={"engine": "none"}, queue={"scale_up_after": "0s", "max_machines": 2, "tick": "5s"})
    return Env(tmp_path, cfg, monkeypatch)


# -- submit ----------------------------------------------------------------


def test_submit_returns_an_id_at_once_and_queues_the_bundle(env):
    out = env.data("queue", "submit", str(env.job_dir(timeout="30m")), "--env", "A=1")
    job_id = out["job_id"]
    assert job_id.startswith("q-") and len(job_id) <= 60
    assert out["paused"] is False and out["state"] == "queued"
    job = env.backend.get(job_id)
    assert job.state == JobState.QUEUED and job.submitted_by == "alice"
    assert job.manifest["run"] == "bash run.sh" and job.manifest["env"] == {"A": "1"}
    assert job.manifest["timeout"] == "30m"
    assert Path(job.bundle_ref).is_file()
    assert env.events("queue_job_submitted")[0]["job_id"] == job_id
    assert api.submit(env.job_dir(), project=env.path).startswith("q-")


def test_submit_rejects_bad_jobs_and_stores_nothing(env, monkeypatch):
    env.cli("queue", "submit", str(env.job_dir("norun", run=None)), expect=2)
    env.cli("queue", "submit", str(env.job_dir("badtimeout", timeout="soon")), expect=2)
    env.cli("queue", "submit", str(env.tmp / "missing"), expect=2)
    monkeypatch.setattr(jobmod, "BUNDLE_MAX_BYTES", 10)
    big = env.job_dir("big")
    (big / "data.bin").write_bytes(__import__("os").urandom(4096))
    assert env.cli("queue", "submit", str(big), expect=None).exit_code != 0
    assert env.backend.list() == []
    assert not list((env.cfg.state_dir / "queue" / "bundles").glob("*"))


def test_two_submits_with_one_key_leave_one_job_and_one_bundle(env, monkeypatch):
    d = env.job_dir()
    barrier = threading.Barrier(4)
    real = LocalOutputStore.put_bundle

    def synced(self, job_id, path):
        barrier.wait(10)  # every submit has passed the "known key" check by now
        return real(self, job_id, path)

    monkeypatch.setattr(LocalOutputStore, "put_bundle", synced)
    ids, errors = [], []

    def go():
        try:
            ids.append(api.submit(d, key="same", project=env.path))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == [] and len(set(ids)) == 1 and len(env.backend.list()) == 1
    assert [p.name for p in (env.cfg.state_dir / "queue" / "bundles").glob("*")] == [f"{ids[0]}.tar.gz"]


def test_a_failed_submit_leaves_no_bundle(env, monkeypatch):
    def boom(job):
        raise RuntimeError("backend down")

    monkeypatch.setattr(env.backend, "submit", boom)
    assert env.cli("queue", "submit", str(env.job_dir()), expect=None).exit_code != 0
    assert not list((env.cfg.state_dir / "queue" / "bundles").glob("*"))


def test_env_values_never_appear_in_job_descriptions(env):
    job_id = env.data("queue", "submit", str(env.job_dir()), "--env", "TOKEN=s3cr3t-value")["job_id"]
    for out in (env.cli("queue", "status", job_id).stdout, env.cli("queue", "cancel", job_id).stdout):
        assert "s3cr3t-value" not in out
        assert json.loads(out)["manifest"]["env"] == {"TOKEN": "***"}
    assert env.backend.get(job_id).manifest["env"] == {"TOKEN": "s3cr3t-value"}  # the job itself still gets it
    assert "s3cr3t-value" not in json.dumps(api.status(job_id, env.path))


def test_submit_while_paused_queues_and_says_so(env):
    env.cli("queue", "pause")
    out = env.data("queue", "submit", str(env.job_dir()))
    assert out["paused"] is True
    assert env.backend.get(out["job_id"]).state == JobState.QUEUED


def test_submit_is_idempotent_with_a_key(env):
    d = env.job_dir()
    a = env.data("queue", "submit", str(d), "--key", "nightly-1")
    b = env.data("queue", "submit", str(d), "--key", "nightly-1")
    assert a["job_id"] == b["job_id"] and b["existing"] is True
    assert a["job_id"] != "nightly-1" and a["key"] == "nightly-1"
    assert len(env.backend.list()) == 1
    assert len(list((env.cfg.state_dir / "queue" / "bundles").glob("*"))) == 1


# -- status ---------------------------------------------------------------


def test_status_summary_has_pool_spend_and_coordinator(env):
    env.cli("queue", "submit", str(env.job_dir()))
    env.pool.open()
    env.pool.poll()
    env.pool.poll()
    env.backend.acquire_coordinator("coord-x", 60)
    s = env.data("queue", "status")
    assert s["queue"]["by_state"]["queued"] == 1
    assert s["coordinator"]["owner"] == "coord-x" and s["coordinator"]["lease_remaining_s"] > 0
    assert s["paused"] is False
    assert s["pool"]["usd_per_hour"] == pytest.approx(1.0)
    assert [(m["slot"], m["state"], m["hourly_usd"]) for m in s["pool"]["machines"]] == [("s1", "ready", 1.0)]
    env.cli("queue", "pause")
    assert env.data("queue", "status")["paused"] is True
    human = runner.invoke(app, ["--config", str(env.path), "queue", "status"])
    assert human.exit_code == 0 and "PAUSED" in human.output and "coord-x" in human.output


def test_status_of_one_job_and_an_unknown_job(env):
    job_id = env.data("queue", "submit", str(env.job_dir()))["job_id"]
    info = env.data("queue", "status", job_id)
    assert info["id"] == job_id and info["state"] == "queued" and info["final"] is False
    env.cli("queue", "status", "q-nope", expect=3)
    assert [j["id"] for j in env.data("queue", "ls")["jobs"]] == [job_id]
    assert env.data("queue", "ls", "--state", "failed")["jobs"] == []


# -- controls -------------------------------------------------------------


def test_control_actions_are_audited_with_the_actor(env):
    job_id = env.data("queue", "submit", str(env.job_dir()))["job_id"]
    env.cli("queue", "pause")
    env.cli("queue", "resume")
    env.cli("queue", "cancel", job_id)
    env.cli("queue", "drain")
    # a failed job to retry
    other = env.data("queue", "submit", str(env.job_dir()))["job_id"]
    claim = env.backend.claim("w", 60, job_id=other) or env.backend.claim("w", 60)
    env.backend.settle(claim, JobState.FAILED, {"reason": "x"})
    env.cli("queue", "retry", claim.job.id)
    rows = [(r["actor"], r["action"]) for r in env.backend.audit()]
    assert rows == [
        ("alice", "set_paused"), ("alice", "set_paused"), ("alice", "request_cancel"),
        ("alice", "set_paused"), ("alice", "retry"),
    ]
    logged = [(e["action"], e["actor"]) for e in env.events("queue_control")]
    assert logged == [
        ("pause", "alice"), ("resume", "alice"), ("cancel", "alice"), ("drain", "alice"), ("retry", "alice"),
    ]
    env.cli("queue", "retry", job_id, expect=3)  # cancelled, not failed
    assert all(e["ok"] is True for e in env.events("queue_control")[:5])
    failed = env.events("queue_control")[-1]
    assert (failed["action"], failed["ok"], failed["error"]) == ("retry", False, "StateError")


def test_drain_now_leaves_the_work_to_the_coordinator_that_holds_the_lease(env):
    env.cli("queue", "submit", str(env.job_dir()))
    env.pool.script(next(iter([j.id for j in env.backend.list()])), FakeRun(polls=50))
    coord = env.coordinator()
    job_id = env.backend.list()[0].id
    env.until(coord, lambda: env.backend.get(job_id).state == JobState.RUNNING)

    out = env.data("queue", "drain", "--now")
    assert out["drain"] == "requested" and out["performed_by"] == "coordinator"
    assert env.backend.drain_requested() is True and env.backend.paused() is True
    assert env.backend.get(job_id).state == JobState.RUNNING  # nothing yet: the coordinator acts on its tick
    assert ("alice", "request_drain") in [(r["actor"], r["action"]) for r in env.backend.audit()]

    env.step(coord)
    job = env.backend.get(job_id)
    assert job.state == JobState.QUEUED and job.cancel_requested is False
    assert [a.outcome.value for a in job.attempts] == ["drained"]
    assert env.pool.retired and all(m.state in ("retired", "gone") for m in env.pool.machines())
    env.step(coord)
    assert env.backend.drain_requested() is False
    assert env.backend.paused() is True
    assert env.pool.opens == 1  # nothing was opened while paused and draining


def test_drain_now_without_a_coordinator_does_it_directly(env):
    env.cli("queue", "submit", str(env.job_dir()))
    job_id = env.backend.list()[0].id
    env.pool.script(job_id, FakeRun(polls=50))
    coord = env.coordinator()
    env.until(coord, lambda: env.backend.get(job_id).state == JobState.RUNNING)
    coord.release()  # the coordinator went away; its job lease runs out
    env.clock.advance(1000)

    out = env.data("queue", "drain", "--now")

    assert out["drain"] == "done" and out["performed"] is True
    job = env.backend.get(job_id)
    assert job.state == JobState.QUEUED
    assert [a.outcome.value for a in job.attempts] == ["drained"]
    assert all(m.state in ("retired", "gone") for m in env.pool.machines())
    assert env.backend.paused() is True and env.backend.drain_requested() is False
    assert env.backend.coordinator() is None  # the lease it took is given back
    logged = env.events("queue_control")
    assert [e["action"] for e in logged] == ["drain_now_done", "drain_now"]  # recorded once it happened
    assert logged[-1]["ok"] is True and logged[-1]["actor"] == "alice"


def test_drain_now_does_not_burn_the_last_attempt(env):
    env.cfg.queue.max_attempts = 1
    env.cli("queue", "submit", str(env.job_dir()))
    job_id = env.backend.list()[0].id
    env.pool.script(job_id, FakeRun(polls=50), FakeRun())
    coord = env.coordinator()
    env.until(coord, lambda: env.backend.get(job_id).state == JobState.RUNNING)
    coord.drain_now("alice")
    assert env.backend.get(job_id).state == JobState.QUEUED
    env.cli("queue", "resume")
    env.until(coord, lambda: env.backend.get(job_id).state == JobState.SUCCEEDED)  # the one real attempt is still its own


def test_drain_now_keeps_a_run_whose_machine_will_not_retire(env):
    env.cli("queue", "submit", str(env.job_dir()))
    job_id = env.backend.list()[0].id
    env.pool.script(job_id, FakeRun(polls=50), FakeRun())
    coord = env.coordinator()
    env.until(coord, lambda: env.backend.get(job_id).state == JobState.RUNNING)
    slot = env.pool.slot_of(f"{job_id}-a1")
    env.pool.retire_refused.add(slot)

    env.data("queue", "drain", "--now")
    env.step(coord, 3)
    job = env.backend.get(job_id)
    assert job.state == JobState.RUNNING and job.attempts == []  # not requeued while its machine lives
    assert env.backend.drain_requested() is True

    env.pool.retire_refused.clear()
    env.step(coord, 3)
    job = env.backend.get(job_id)
    assert job.state == JobState.QUEUED and [a.outcome.value for a in job.attempts] == ["drained"]
    assert env.backend.drain_requested() is False


def test_direct_drain_now_waits_for_a_job_it_has_not_adopted(env):
    env.cli("queue", "submit", str(env.job_dir()))
    job_id = env.backend.list()[0].id
    env.pool.script(job_id, FakeRun(polls=50))
    coord = env.coordinator()
    env.until(coord, lambda: env.backend.get(job_id).state == JobState.RUNNING)
    coord.release()  # the coordinator went away but the job's lease is still alive

    out = env.data("queue", "drain", "--now")

    assert out["drain"] == "partial" and out["waiting_for"] == [job_id]
    assert [m for m in env.pool.machines() if m.state not in ("retired", "gone")]  # its machine was left alone
    assert env.backend.get(job_id).state == JobState.RUNNING
    assert env.backend.drain_requested() is True  # the next coordinator finishes it


def test_resume_drops_a_pending_drain_and_a_pending_drain_pauses_once(env):
    env.backend.acquire_coordinator("someone", 600)
    env.cli("queue", "drain", "--now")
    assert env.backend.drain_requested() is True
    assert [r["action"] for r in env.backend.audit()] == ["set_paused", "request_drain"]
    out = env.data("queue", "resume")
    assert out["drain_cleared"] is True
    assert env.backend.drain_requested() is False and env.backend.paused() is False


def test_a_paused_queue_keeps_no_warm_machine(write_config, tmp_path, monkeypatch):
    cfg = write_config(runtime={"engine": "none"}, queue={"scale_up_after": "0s", "min_machines": 1, "tick": "5s"})
    e = Env(tmp_path, cfg, monkeypatch)
    e.backend.set_paused(True, "alice")
    coord = e.coordinator()
    e.step(coord, 3)
    assert e.pool.opens == 0
    e.backend.set_paused(False, "alice")
    e.step(coord, 2)
    assert e.pool.opens == 1


# -- the coordinator from the CLI -------------------------------------------


def test_run_once_ticks_and_gives_the_lease_back(env):
    env.cli("queue", "submit", str(env.job_dir()))
    out = env.data("queue", "run", "--once")
    assert out["leader"] is True and out["opened"] == 1  # a machine for the queued job
    assert env.backend.coordinator() is None


def test_metrics_are_served_on_loopback_only(env):
    env.obs.gauge("gpu_queue_depth", 3)
    server = api.serve_metrics(env.obs, 0)
    try:
        host, port = server.server_address[:2]
        assert host == "127.0.0.1"
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
        assert "gpu_queue_depth 3" in body
        with pytest.raises(Exception):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/other", timeout=5)
    finally:
        server.shutdown()
        server.server_close()


def test_stats_reads_what_the_coordinator_wrote(env):
    assert env.data("queue", "stats")["metrics"] == {}
    env.cli("queue", "submit", str(env.job_dir()))
    env.data("queue", "run", "--once")
    out = env.data("queue", "stats")
    assert out["metrics"]["gpu_queue_coordinator_leader"] == 1.0
    assert out["queue"]["by_state"]


def test_audit_exits_1_while_something_stays_unrepaired(env, monkeypatch):
    from gpu4agents.queue.audit import Violation

    monkeypatch.setattr("gpu4agents.queue.audit.audit", lambda *a, **k: [Violation("I7", None, None, "x", False)])
    result = env.cli("queue", "audit", "--repair", expect=1)
    assert json.loads(result.stdout)["violations"][0]["code"] == "I7"
    monkeypatch.setattr("gpu4agents.queue.audit.audit", lambda *a, **k: [Violation("I4", None, "s1", "x", True)])
    env.cli("queue", "audit", "--repair", expect=0)


def test_an_unreachable_backend_is_one_clean_error(env, monkeypatch):
    def down(*a, **k):
        raise ConnectionError("Error 111 connecting to redis://u:hunter2@host:6379. Connection refused.")

    monkeypatch.setattr(env.backend, "stats", down)
    result = env.cli("queue", "status", expect=4)
    assert "hunter2" not in result.output and "unreachable" in result.output and result.exception.__class__.__name__ == "SystemExit"


def test_run_once_gives_the_lease_back_even_when_the_tick_fails_and_a_busy_port_is_a_config_error(env, monkeypatch):
    def boom(self):
        self.leader = True
        self.backend.acquire_coordinator(self.worker_id, 60)
        raise RuntimeError("tick failed")

    monkeypatch.setattr(Coordinator, "tick", boom)
    env.cli("queue", "run", "--once", expect=None)
    assert env.backend.coordinator() is None
    monkeypatch.undo()
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        env.cli("queue", "run", "--once", "--metrics-port", str(port), expect=2)


def test_audit_reports_no_violations_on_a_clean_queue(env):
    out = env.data("queue", "audit")
    assert out["ok"] is True and out["violations"] == []


def test_queue_run_is_quiet_about_stopped_billing_but_other_queue_commands_are_not(env, monkeypatch):
    calls = []
    monkeypatch.setattr(_common, "warn_stopped_billing", lambda: calls.append(1))
    base = ["--config", str(env.path), "queue"]
    runner.invoke(app, [*base, "run", "--once"])
    assert calls == []
    runner.invoke(app, [*base, "ls"])
    assert calls == [1]


def test_without_a_queue_block_it_is_a_config_error(write_config, tmp_path, monkeypatch):
    write_config(runtime={"engine": "none"})
    result = runner.invoke(app, ["--json", "--config", str(tmp_path / "gpu.yaml"), "queue", "ls"])
    assert result.exit_code == 2


# -- the Python API ----------------------------------------------------------


def test_pull_gets_verified_outputs_after_the_job_ran(env):
    job_id = env.data("queue", "submit", str(env.job_dir()))["job_id"]
    env.cli("queue", "pull", job_id, expect=3)  # no outputs yet
    env.pool.script(job_id, FakeRun(outputs={"model.bin": b"weights"}))
    coord = env.coordinator()
    env.until(coord, lambda: env.backend.get(job_id).state == JobState.SUCCEEDED)
    out = env.data("queue", "pull", job_id)
    assert (Path(out["path"]) / "model.bin").read_bytes() == b"weights"
    assert out["files"] == ["model.bin"]
    dest = env.tmp / "elsewhere"
    out = env.data("queue", "pull", job_id, "--to", str(dest))
    assert (dest / "model.bin").read_bytes() == b"weights" and "model.bin" in out["files"]


def test_python_api_submit_and_wait(env):
    job_id = api.submit(env.job_dir(), key="k1", project=env.path)
    env.pool.script(job_id, FakeRun(outputs={"result.txt": b"42"}))
    stop = threading.Event()
    coord = Coordinator(env.cfg, env.backend, env.outputs, env.pool, env.obs, worker_id="thread-coord")

    def loop() -> None:
        while not stop.is_set():
            coord.tick()
            time.sleep(0.02)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        monkey_poll = api.WAIT_POLL_S
        api.WAIT_POLL_S = 0.05
        try:
            done = api.wait(job_id, timeout_s=30, project=env.path)
        finally:
            api.WAIT_POLL_S = monkey_poll
        assert done["state"] == "succeeded" and done["final"] is True
        assert api.status(job_id, env.path)["state"] == "succeeded"
        path = api.outputs(job_id, project=env.path)
        assert (path / "result.txt").read_bytes() == b"42"
        assert api.cancel(job_id, env.path)["state"] == "succeeded"
    finally:
        stop.set()
        thread.join(5)


def test_wait_times_out_with_the_current_state(env):
    job_id = api.submit(env.job_dir(), project=env.path)
    old = api.WAIT_POLL_S
    api.WAIT_POLL_S = 0.05
    try:
        got = api.wait(job_id, timeout_s=0.2, project=env.path)
    finally:
        api.WAIT_POLL_S = old
    assert got["timed_out"] is True and got["state"] == "queued"


# -- the service ---------------------------------------------------------------


def test_queue_service_unit_runs_gpu_queue_run(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(timer, "_platform", lambda: "linux")
    project = tmp_path / "my project%1"
    project.mkdir()
    text = timer.render_queue_unit(python="/opt/py thon/bin/python", project_dir=project, jambu_home="/h/100%")
    assert f"WorkingDirectory={project}".replace("%", "%%") in text
    assert 'ExecStart="/opt/py thon/bin/python" -m gpu4agents queue run' in text
    assert "Restart=on-failure" in text and "WantedBy=default.target" in text
    assert 'Environment="GPU_HOME=/h/100%%"' in text
    plist = timer.render_queue_plist(python="/usr/bin/python3", project_dir=project)
    assert "<string>queue</string>" in plist and "<key>SuccessfulExit</key><false/>" in plist
    assert f"<key>WorkingDirectory</key><string>{project}</string>" in plist

    done = timer.install_queue_service(project, python="/usr/bin/python3")
    path = Path(done["files"][0])
    assert path.is_file() and path.parent == tmp_path / "cfg" / "systemd" / "user"
    assert done["enable"].startswith("systemctl --user daemon-reload") and done["name"] in done["enable"]
    removed = timer.uninstall_queue_service(project)
    assert removed["removed"] == [str(path)] and not path.exists()


def test_install_service_prints_the_enable_command_and_does_not_enable(env, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(timer, "_platform", lambda: "linux")
    out = env.data("queue", "install-service")
    assert "systemctl --user" in out["enable"] and Path(out["files"][0]).is_file()
    out = env.data("queue", "install-service", "--uninstall")
    assert out["removed"] and not Path(out["removed"][0]).exists()


# -- MCP ---------------------------------------------------------------------


def _call(server, name, **arguments):
    result = server.call_tool(name, arguments)
    return result["isError"], json.loads(result["content"][0]["text"])


def test_queue_tools_are_listed_with_the_right_annotations(tmp_path):
    tools = {t["name"]: t for t in Server(project=tmp_path).handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    for name in ("gpu_queue_submit", "gpu_queue_status", "gpu_queue_cancel", "gpu_queue_pull", "gpu_queue_control"):
        assert name in tools
    assert tools["gpu_queue_status"]["annotations"]["readOnlyHint"] is True
    assert tools["gpu_queue_control"]["annotations"]["destructiveHint"] is True
    assert tools["gpu_queue_submit"]["annotations"]["idempotentHint"] is False
    assert tools["gpu_queue_control"]["inputSchema"]["properties"]["action"]["enum"] == ["pause", "resume", "drain"]


def test_drain_now_requires_confirm_in_mcp(env, monkeypatch):
    server = Server(project=env.tmp)
    calls = []
    monkeypatch.setattr(server, "cli", lambda argv, cwd, timeout=0: calls.append(argv))
    is_error, payload = _call(server, "gpu_queue_control", action="drain", now=True)
    assert is_error is True and "confirm=true" in payload["error"] and calls == []
    is_error, _ = _call(server, "gpu_queue_control", action="pause", now=True)
    assert is_error is True and calls == []
    from gpu4agents.agent.mcp import CliResult

    monkeypatch.setattr(server, "cli", lambda argv, cwd, timeout=0: calls.append(argv) or CliResult(0, "{}", "", {}))
    is_error, _ = _call(server, "gpu_queue_control", action="drain", now=True, confirm=True)
    assert is_error is False and calls == [["queue", "drain", "--now"]]


def test_mcp_submit_and_status_call_the_cli(env):
    server = Server(project=env.tmp, python=__import__("sys").executable)
    # the subprocess cannot see the in-process fakes, so only the argv shape is checked
    seen = []
    server.cli = lambda argv, cwd, timeout=0: seen.append(argv) or __import__("gpu4agents.agent.mcp", fromlist=["x"]).CliResult(0, "{}", "", {"job_id": "q-1"})
    is_error, data = _call(server, "gpu_queue_submit", dir=str(env.tmp / "job"), key="k", env={"A": "1"})
    assert is_error is False and data["job_id"] == "q-1" and "next" in data
    assert seen[0] == ["queue", "submit", str(env.tmp / "job"), "--key", "k", "--env", "A=1"]
    _call(server, "gpu_queue_status", job_id="q-1")
    _call(server, "gpu_queue_pull", job_id="q-1", dest="/x")
    _call(server, "gpu_queue_cancel", job_id="q-1")
    assert seen[1:] == [
        ["queue", "status", "q-1"], ["queue", "pull", "q-1", "--to", "/x"], ["queue", "cancel", "q-1"],
    ]
