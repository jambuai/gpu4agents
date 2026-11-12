"""The coordinator: claims, preflight, deadline, dispatch, verified delivery, retries
on machine faults, takeover and recovery after a crash.

A real LocalBackend and LocalOutputStore with an injected clock; the machines are a
FakePool with scripted outcomes (tests/queue_fakes.py).
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from gpu4agents.core import events as ev
from gpu4agents.core.events import EventLog
from gpu4agents.queue.backends.local import LocalBackend
from gpu4agents.queue.coordinator import Coordinator, TickReport
from gpu4agents.queue.models import JobState, Outcome, QueueJob
from gpu4agents.queue.observe import Observer
from gpu4agents.queue.outputs import LocalOutputStore, OutputRef

from queue_fakes import FakePool, FakeRun

TICK_S = 5.0


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class World:
    def __init__(self, tmp_path: Path, cfg, pool_kwargs: dict) -> None:
        self.tmp = tmp_path
        self.cfg = cfg
        self.clock = Clock()
        self.backend = LocalBackend(tmp_path / "queue.db", clock=self.clock)
        self.outputs = LocalOutputStore(tmp_path / "gpu-outputs", bundles_dir=tmp_path / "bundles")
        self.obs = Observer(EventLog(tmp_path / "logs"), tmp_path / "metrics.prom")
        self.pool = FakePool(self.obs, self.clock, tmp_path / "machines", **pool_kwargs)

    def coordinator(self, worker_id: str = "coord-a", **kwargs) -> Coordinator:
        return Coordinator(
            self.cfg, self.backend, self.outputs, self.pool, self.obs,
            worker_id=worker_id, clock=self.clock, **kwargs,
        )

    def submit(self, job_id: str, **manifest) -> QueueJob:
        src = self.tmp / "src" / f"{job_id}.tar.gz"
        src.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(src, "w:gz") as tar:
            data = b"echo hello\n"
            info = tarfile.TarInfo("run.sh")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        ref = self.outputs.put_bundle(job_id, src)
        man = {"run": "bash run.sh", "timeout": "10m", **manifest}
        return self.backend.submit(
            QueueJob(id=job_id, key=f"key-{job_id}", bundle_ref=ref, manifest=man,
                     submitted_at=0.0, submitted_by="tester")
        )

    def job(self, job_id: str) -> QueueJob:
        job = self.backend.get(job_id)
        assert job is not None
        return job

    def step(self, coord: Coordinator, ticks: int = 1) -> list[TickReport]:
        out = []
        for _ in range(ticks):
            self.clock.advance(TICK_S)
            out.append(coord.tick())
        return out

    def until(self, coord: Coordinator, pred, max_ticks: int = 40) -> None:
        for _ in range(max_ticks):
            if pred():
                return
            self.step(coord)
        assert pred(), "condition not reached"

    def until_state(self, coord: Coordinator, job_id: str, *states: JobState, max_ticks: int = 40) -> QueueJob:
        self.until(coord, lambda: self.job(job_id).state in states, max_ticks)
        return self.job(job_id)

    def final(self, coord: Coordinator, job_id: str, max_ticks: int = 40) -> QueueJob:
        return self.until_state(
            coord, job_id, JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, max_ticks=max_ticks
        )

    def events(self, name: str | None = None) -> list[dict]:
        path = self.tmp / "logs" / "events.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [r for r in rows if name is None or r["event"] == name]


@pytest.fixture
def world(write_config, tmp_path):
    def make(pool: dict | None = None, **queue) -> World:
        q = {"scale_up_after": "0s", "max_machines": 2, "tick": f"{TICK_S:g}s", **queue}
        cfg = write_config(runtime={"engine": "none"}, queue=q)
        return World(tmp_path, cfg, dict(pool or {}))

    return make


def test_a_job_runs_and_its_outputs_are_delivered_and_verified(world):
    w = world()
    w.pool.script("j1", FakeRun(outputs={"model.bin": b"weights", "logs/train.txt": b"loss 0.1"}))
    w.submit("j1")
    coord = w.coordinator()

    job = w.final(coord, "j1")

    assert job.state == JobState.SUCCEEDED
    ref = OutputRef(**job.outputs_ref)
    assert w.outputs.verify(ref) == []
    assert (Path(ref.location) / "model.bin").read_bytes() == b"weights"
    assert (Path(ref.location) / "logs" / "train.txt").read_bytes() == b"loss 0.1"
    assert [(a.number, a.remote_id, a.outcome, a.exit_code) for a in job.attempts] == [
        (1, "j1-a1", Outcome.SUCCEEDED, 0)
    ]
    assert job.result["outcome"] == "succeeded"
    assert job.result["outputs"] == ref.location


def test_jobs_run_back_to_back_on_one_warm_machine(world):
    w = world(max_machines=1)
    for j in ("j1", "j2", "j3"):
        w.submit(j)
    coord = w.coordinator()

    for j in ("j1", "j2", "j3"):
        assert w.final(coord, j).state == JobState.SUCCEEDED

    assert w.pool.opens == 1
    assert [r.id for r in w.pool.starts] == ["j1-a1", "j2-a1", "j3-a1"]
    assert {r.slot for r in w.pool.starts} == {"s1"}


def test_the_deadline_moves_with_each_job(world):
    w = world(max_machines=1, defaults={"deadline_margin": "5m"})
    w.submit("j1", timeout="10m")
    w.submit("j2", timeout="30m")
    coord = w.coordinator()

    w.final(coord, "j1")
    w.final(coord, "j2")

    assert len(w.pool.deadlines) == 2
    (s1, until1, at1), (s2, until2, at2) = w.pool.deadlines
    assert s1 == s2 == "s1"
    assert until1 == pytest.approx(at1 + 600 + 300)
    assert until2 == pytest.approx(at2 + 1800 + 300)
    assert until2 > until1
    m = w.pool.machines()[0]
    assert m.deadline_at == pytest.approx(until2)


def test_a_lost_machine_requeues_the_job_on_another(world):
    w = world(max_machines=2)
    w.pool.script("j1", FakeRun(polls=50), FakeRun())
    w.submit("j1")
    coord = w.coordinator()
    w.until_state(coord, "j1", JobState.RUNNING)
    first = w.pool.slot_of("j1-a1")

    w.pool.lose(first)
    job = w.final(coord, "j1")

    assert job.state == JobState.SUCCEEDED
    assert [a.outcome for a in job.attempts] == [Outcome.MACHINE_LOST, Outcome.SUCCEEDED]
    assert job.attempts[0].machine == first
    assert job.attempts[1].machine != first
    assert [(s, b) for s, _, b in w.pool.retired] == [(first, False)]
    assert [e["cause"] for e in w.events(ev.QUEUE_JOB_RETRIED)] == ["machine_lost"]


def test_a_crash_and_failed_health_check_count_as_the_machine(world):
    w = world(max_machines=2)
    w.pool.script("j1", FakeRun(exit_code=139), FakeRun())
    w.submit("j1")
    coord = w.coordinator()
    w.until_state(coord, "j1", JobState.RUNNING)
    first = w.pool.slot_of("j1-a1")
    w.pool.health_failures.append(first)

    job = w.final(coord, "j1")

    assert job.state == JobState.SUCCEEDED
    assert [a.outcome for a in job.attempts] == [Outcome.UNHEALTHY, Outcome.SUCCEEDED]
    assert job.attempts[0].exit_code == 139
    assert [(s, b) for s, _, b in w.pool.retired] == [(first, True)]
    assert job.attempts[1].machine != first


def test_a_plain_failure_is_the_job_s_and_is_not_retried(world):
    w = world()
    w.pool.script("j1", FakeRun(exit_code=3, outputs={"err.txt": b"bad input"}))
    w.submit("j1")
    coord = w.coordinator()

    job = w.final(coord, "j1")

    assert job.state == JobState.FAILED
    assert [(a.outcome, a.exit_code) for a in job.attempts] == [(Outcome.JOB_FAILED, 3)]
    assert len(w.pool.starts) == 1
    assert w.pool.retired == []
    assert job.result["outcome"] == "job_failed"
    # the failed attempt's outputs are kept, verified, apart from a success
    ref = OutputRef(**job.outputs_ref)
    assert w.outputs.verify(ref) == []
    assert (Path(ref.location) / "err.txt").read_bytes() == b"bad input"
    assert w.events(ev.QUEUE_JOB_FAILED)
    assert not w.events(ev.QUEUE_JOB_RETRIED)


def test_max_attempts_ends_in_failed_with_every_reason(world):
    w = world(max_machines=3, max_attempts=2)
    w.pool.script("j1", FakeRun(polls=50), FakeRun(polls=50), FakeRun())
    w.submit("j1")
    coord = w.coordinator()
    w.until_state(coord, "j1", JobState.RUNNING)
    w.pool.lose(w.pool.slot_of("j1-a1"))
    w.until(coord, lambda: "j1-a2" in w.pool.remotes and w.job("j1").state == JobState.RUNNING)
    w.pool.hang(w.pool.slot_of("j1-a2"))

    job = w.final(coord, "j1")

    assert job.state == JobState.FAILED
    assert [a.outcome for a in job.attempts] == [Outcome.MACHINE_LOST, Outcome.UNRESPONSIVE]
    assert len(w.pool.starts) == 2
    assert "machine_lost" in job.result["reason"] and "unresponsive" in job.result["reason"]
    assert [a["outcome"] for a in job.result["attempts"]] == ["machine_lost", "unresponsive"]
    dead = w.events(ev.QUEUE_JOB_DEAD)
    assert len(dead) == 1 and dead[0]["job_id"] == "j1"


def test_retry_gets_the_previous_outputs(world):
    w = world(max_machines=2)
    w.pool.script("j1", FakeRun(exit_code=1, outputs={"ckpt/step-5.pt": b"step 5"}), FakeRun())
    w.submit("j1")
    coord = w.coordinator()
    w.until_state(coord, "j1", JobState.RUNNING)
    w.pool.health_failures.append(w.pool.slot_of("j1-a1"))

    job = w.final(coord, "j1")

    assert job.state == JobState.SUCCEEDED
    first, second = w.pool.starts
    assert "JAMBU_PREVIOUS_OUTPUT_DIR" not in first.env
    prev = second.env["JAMBU_PREVIOUS_OUTPUT_DIR"]
    assert prev.endswith("/j1-a2/src/.jambu-previous")
    # what the job sees on the machine: its own code plus attempt 1's files
    src = second.workdir / "src"
    assert (src / "run.sh").read_bytes() == b"echo hello\n"
    assert (src / ".jambu-previous" / "ckpt" / "step-5.pt").read_bytes() == b"step 5"
    # the failed attempt's outputs are stored apart and verify
    assert any(p.name == ".attempt-1" for p in Path(job.outputs_ref["location"]).iterdir())


def test_preflight_rejection_settles_failed_with_the_reason(world):
    w = world(pool={"vram_gb": 24.0})
    w.submit("j1", min_vram_gb=80)
    coord = w.coordinator()

    job = w.final(coord, "j1")

    assert job.state == JobState.FAILED
    assert [a.outcome for a in job.attempts] == [Outcome.PREFLIGHT_REJECTED]
    assert "80" in job.result["reason"] and "24" in job.result["reason"]
    assert job.result["outcome"] == "preflight_rejected"
    assert w.pool.starts == []
    rejected = w.events(ev.QUEUE_PREFLIGHT_REJECTED)
    assert len(rejected) == 1 and rejected[0]["job_id"] == "j1"


def test_a_custom_preflight_is_used(world):
    from gpu4agents.queue.preflight import PreflightResult

    seen = []

    def refuse(job, ctx):
        seen.append((job.id, ctx.attempt, ctx.hourly_usd))
        return PreflightResult(False, None, "not today")

    w = world()
    w.submit("j1")
    coord = w.coordinator(preflight=refuse)

    job = w.final(coord, "j1")

    assert job.result["reason"] == "not today"
    assert seen == [("j1", 1, 1.0)]


def test_crash_after_dispatch_adopts_the_running_job_instead_of_starting_it_twice(world):
    w = world()
    w.pool.script("j1", FakeRun(polls=40))
    w.submit("j1")
    first = w.coordinator("coord-a")
    w.until_state(first, "j1", JobState.RUNNING)
    assert len(w.pool.starts) == 1
    del first  # killed: no release, no settle

    w.clock.advance(120)  # every lease the dead coordinator held has expired
    second = w.coordinator("coord-b")
    job = w.final(second, "j1", max_ticks=80)

    assert job.state == JobState.SUCCEEDED
    assert len(w.pool.starts) == 1
    assert [a.remote_id for a in job.attempts] == ["j1-a1"]
    assert [d[2] for d in w.pool.dispatches] == ["started"]  # adopted, not dispatched again
    assert w.events(ev.COORDINATOR_LEASE_ACQUIRED)[-1]["worker"] == "coord-b"


def test_a_job_whose_coordinator_died_mid_run_is_adopted_while_still_running(world):
    w = world()
    w.pool.script("j1", FakeRun(polls=200))
    w.submit("j1")
    first = w.coordinator("coord-a")
    w.until_state(first, "j1", JobState.RUNNING)
    del first

    second = w.coordinator("coord-b")
    w.until(
        second,
        lambda: w.job("j1").state == JobState.RUNNING and (w.job("j1").limits or {}).get("adopted_by") == "coord-b",
        max_ticks=100,
    )
    # finish it now
    w.pool.remotes["j1-a1"].polls_left = 1
    job = w.final(second, "j1")

    assert job.state == JobState.SUCCEEDED
    assert len(w.pool.starts) == 1
    assert [d[2] for d in w.pool.dispatches] == ["started"]


def test_a_job_lost_with_its_coordinator_and_machine_is_requeued(world):
    w = world()
    w.pool.script("j1", FakeRun(polls=100), FakeRun())
    w.submit("j1")
    first = w.coordinator("coord-a")
    w.until_state(first, "j1", JobState.RUNNING)
    w.pool.lose(w.pool.slot_of("j1-a1"))
    del first

    second = w.coordinator("coord-b")
    job = w.final(second, "j1", max_ticks=100)

    assert job.state == JobState.SUCCEEDED
    assert [(a.remote_id, a.outcome) for a in job.attempts] == [
        ("j1-a1", Outcome.MACHINE_LOST), ("j1-a2", Outcome.SUCCEEDED)
    ]


class Crash(BaseException):
    """The process dies here."""


def test_a_crash_between_recording_and_settling_is_finished_not_run_again(world, monkeypatch):
    w = world()
    w.submit("j1")
    first = w.coordinator("coord-a")
    real = w.backend.settle

    def die(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(w.backend, "settle", die)
    with pytest.raises(Crash):
        w.until(first, lambda: False, max_ticks=20)
    monkeypatch.setattr(w.backend, "settle", real)
    del first
    assert w.job("j1").attempts[0].outcome == Outcome.SUCCEEDED  # recorded, not settled

    second = w.coordinator("coord-b")
    job = w.final(second, "j1", max_ticks=100)

    assert job.state == JobState.SUCCEEDED
    assert len(w.pool.starts) == 1
    assert len(job.attempts) == 1
    assert w.outputs.verify(OutputRef(**job.outputs_ref)) == []


def test_a_failed_job_retried_by_hand_runs_again(world):
    w = world()
    w.pool.script("j1", FakeRun(exit_code=2), FakeRun())
    w.submit("j1")
    coord = w.coordinator()
    assert w.final(coord, "j1").state == JobState.FAILED

    w.backend.retry("j1", "tester")
    job = w.final(coord, "j1")

    assert job.state == JobState.SUCCEEDED
    assert [a.outcome for a in job.attempts] == [Outcome.JOB_FAILED, Outcome.SUCCEEDED]
    assert len(w.pool.starts) == 2


def test_delivery_failure_keeps_the_job_delivering_and_retries(world):
    w = world()
    w.pool.fetch_failures["j1-a1"] = 2
    w.pool.script("j1", FakeRun(outputs={"out.txt": b"result"}))
    w.submit("j1")
    coord = w.coordinator()

    w.until_state(coord, "j1", JobState.DELIVERING)
    assert w.events(ev.QUEUE_OUTPUTS_DELIVERY_FAILED)
    job = w.final(coord, "j1")

    assert job.state == JobState.SUCCEEDED
    assert w.pool.fetches.count("j1-a1") == 3
    assert len(w.pool.starts) == 1
    assert len(w.events(ev.QUEUE_OUTPUTS_DELIVERY_FAILED)) == 2
    assert w.outputs.verify(OutputRef(**job.outputs_ref)) == []


def test_a_store_that_does_not_verify_keeps_the_job_delivering(world, monkeypatch):
    w = world()
    w.submit("j1")
    coord = w.coordinator()
    real = w.outputs.verify
    bad = {"n": 2}

    def flaky(ref):
        if bad["n"] > 0:
            bad["n"] -= 1
            return ["result.txt: sha256 mismatch"]
        return real(ref)

    monkeypatch.setattr(w.outputs, "verify", flaky)
    job = w.final(coord, "j1")

    assert job.state == JobState.SUCCEEDED
    failed = w.events(ev.QUEUE_OUTPUTS_DELIVERY_FAILED)
    assert len(failed) == 2 and "sha256 mismatch" in failed[0]["problems"][0]


def test_a_standby_coordinator_takes_over_after_the_lease_expires(world):
    w = world()
    a = w.coordinator("coord-a")
    b = w.coordinator("coord-b")
    w.step(a)
    w.clock.advance(1)
    assert b.tick() == TickReport(0, 0, 0, 0, 0, 0)
    assert not b.leader and a.leader

    w.submit("j1")
    w.clock.advance(60)  # a stopped ticking (died); its lease runs out
    job = w.final(b, "j1", max_ticks=100)

    assert job.state == JobState.SUCCEEDED
    owners = [e["worker"] for e in w.events(ev.COORDINATOR_LEASE_ACQUIRED)]
    assert owners == ["coord-a", "coord-b"]
    assert w.backend.coordinator()["owner"] == "coord-b"

    a.tick()  # back from the dead: it is a standby now
    assert not a.leader
    assert w.events(ev.COORDINATOR_LEASE_LOST)


def test_pause_stops_claims_but_not_running_jobs(world):
    w = world(max_machines=1)
    w.pool.script("j1", FakeRun(polls=4))
    w.submit("j1")
    coord = w.coordinator()
    w.until_state(coord, "j1", JobState.RUNNING)

    w.backend.set_paused(True, "tester")
    w.submit("j2")
    assert w.final(coord, "j1").state == JobState.SUCCEEDED
    w.step(coord, 5)
    assert w.job("j2").state == JobState.QUEUED
    assert [r.id for r in w.pool.starts] == ["j1-a1"]

    w.backend.set_paused(False, "tester")
    assert w.final(coord, "j2").state == JobState.SUCCEEDED


def test_cancel_reaches_the_running_job(world):
    w = world()
    w.pool.script("j1", FakeRun(polls=100))
    w.submit("j1")
    coord = w.coordinator()
    w.until_state(coord, "j1", JobState.RUNNING)

    w.backend.request_cancel("j1", "tester")
    job = w.final(coord, "j1", max_ticks=5)

    assert job.state == JobState.CANCELLED
    assert w.pool.cancels == ["j1-a1"]
    assert [a.outcome for a in job.attempts] == [Outcome.CANCELLED]
    assert w.pool.remotes["j1-a1"].state == "cancelled"


def test_a_stale_lease_write_is_dropped_not_retried(world):
    w = world(max_machines=1)
    w.pool.script("j1", FakeRun(polls=100))
    w.submit("j1")
    coord = w.coordinator()
    w.until_state(coord, "j1", JobState.RUNNING)

    # the coordinator stalls past the job's lease; someone else claims the job
    assert w.backend.renew_coordinator("coord-a", 3600)
    w.clock.advance(3000)
    intruder = w.backend.claim("intruder", 3600, job_id="j1")
    assert intruder is not None

    w.step(coord, 4)

    stale = w.events(ev.QUEUE_STALE_WRITE_REFUSED)
    assert len(stale) == 1 and stale[0]["job_id"] == "j1"
    assert w.obs.snapshot()["gpu_queue_stale_writes_total"] == 1
    job = w.job("j1")
    assert job.state == JobState.CLAIMED and job.attempts == []
    assert w.backend.renew(intruder, 60).token == intruder.token
    assert len(w.pool.starts) == 1

    # the machine still runs the job it no longer owns: nothing else is put on it
    w.submit("j2")
    w.step(coord, 3)
    assert w.job("j2").state == JobState.QUEUED
    assert [r.id for r in w.pool.starts] == ["j1-a1"]


def test_every_decision_is_an_event(world):
    w = world()
    w.submit("j1")
    coord = w.coordinator()
    w.final(coord, "j1")

    names = {e["event"] for e in w.events()}
    for name in (
        ev.QUEUE_JOB_CLAIMED, ev.QUEUE_JOB_STARTED, ev.QUEUE_OUTPUTS_STORED,
        ev.QUEUE_JOB_SUCCEEDED, ev.POOL_SCALE_DECISION, ev.POOL_DEADLINE_EXTENDED,
        ev.COORDINATOR_LEASE_ACQUIRED,
    ):
        assert name in names, name
    stored = w.events(ev.QUEUE_OUTPUTS_STORED)[0]
    assert stored["files"] == 1 and stored["bytes"] > 0 and len(stored["sha256"]) == 64
    metrics = (w.tmp / "metrics.prom").read_text()
    assert 'gpu_queue_jobs_total{final_state="succeeded"} 1' in metrics
    assert 'gpu_queue_attempts_total{outcome="succeeded"} 1' in metrics
    assert "gpu_queue_coordinator_heartbeat_age_seconds" in metrics


def test_the_audit_hook_runs_every_twelve_ticks(world):
    w = world()
    coord = w.coordinator()
    calls = []
    coord._audit = lambda: calls.append(1) or ["v"]
    reports = w.step(coord, 24)
    assert len(calls) == 2
    assert sum(r.violations for r in reports) == 2


def test_run_ticks_until_stopped_and_releases_the_lease(world):
    import threading

    w = world(tick="0.01s")
    coord = w.coordinator()
    stop = threading.Event()
    t = threading.Thread(target=coord.run, args=(stop,))
    t.start()
    try:
        for _ in range(200):
            if w.backend.coordinator():
                break
            stop.wait(0.01)
        assert w.backend.coordinator()["owner"] == "coord-a"
    finally:
        stop.set()
        t.join(5)
    assert not t.is_alive()
    assert w.backend.coordinator() is None
    assert w.events(ev.COORDINATOR_STARTED)[0]["config_hash"]
    assert w.events(ev.COORDINATOR_STOPPED)
