"""The auditor: each guarantee of queue mode, broken on purpose, is found, reported
once per change and repaired when that is safe.

A real LocalBackend and LocalOutputStore with an injected clock; the machines are a
FakePool (tests/queue_fakes.py). The World fixture comes from the coordinator tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpu4agents.core import events as ev
from gpu4agents.core.models import Instance
from gpu4agents.queue.audit import AuditMemory, Violation, audit
from gpu4agents.queue.coordinator import DEADLINE_GRACE_S
from gpu4agents.queue.models import JobState, Outcome
from gpu4agents.queue.outputs import OutputRef

from queue_fakes import FakeRun
from test_queue_coordinator import Crash, World, _ghost_in_flight, world  # noqa: F401 - fixture

METRIC = "gpu_queue_invariant_violations_total"


def run_audit(w: World, coord=None, repair: bool = True, memory: AuditMemory | None = None) -> list[Violation]:
    return audit(
        w.backend, w.pool, w.outputs, w.obs, repair=repair, coordinator=coord,
        config=w.cfg, clock=w.clock, memory=memory,
    )


def codes(found: list[Violation]) -> list[str]:
    return [v.code for v in found]


def count(w: World, code: str) -> float:
    return w.obs.snapshot().get(f'{METRIC}{{code="{code}"}}', 0)


def reported(w: World, code: str) -> list[dict]:
    return [e for e in w.events(ev.QUEUE_INVARIANT_VIOLATED) if e["code"] == code]


def ready_machine(w: World):
    m = w.pool.open()
    w.pool.poll()
    return m


def keep_leases(w: World, coord, seconds: float, step: float = 10.0) -> None:
    """Move the clock without ticking (the machines do not move), heartbeats only."""
    left = seconds
    while left > 0:
        w.clock.advance(min(step, left))
        left -= step
        coord.beat()
    assert coord.leader


def running(w: World, coord, job_id: str):
    w.until_state(coord, job_id, JobState.RUNNING)
    return coord.backend.get(job_id)


# ---------------------------------------------------------------------- clean


def test_a_clean_pool_has_no_violations(world):
    w = world()
    w.submit("j1")
    coord = w.coordinator()
    assert w.final(coord, "j1").state == JobState.SUCCEEDED

    assert run_audit(w, coord) == []
    assert run_audit(w) == []
    assert w.events(ev.QUEUE_INVARIANT_VIOLATED) == []
    assert not any(k.startswith(METRIC) for k in w.obs.snapshot())


# ---------------------------------------------------------------------- I1


def test_i1_a_running_job_whose_machine_is_gone_is_requeued_machine_lost(world):
    w = world()
    w.pool.script("j1", FakeRun(polls=500), FakeRun())
    w.submit("j1")
    coord = w.coordinator()
    job = running(w, coord, "j1")
    w.pool.lose(job.machine)

    found = run_audit(w, coord)

    assert codes(found) == ["I1_running_without_machine"]
    assert found[0].repaired and found[0].job_id == "j1" and found[0].slot == job.machine
    job = w.job("j1")
    assert job.state == JobState.QUEUED
    assert [a.outcome for a in job.attempts] == [Outcome.MACHINE_LOST]
    assert count(w, "I1_running_without_machine") == 1
    [event] = reported(w, "I1_running_without_machine")
    assert event["job_id"] == "j1" and event["repaired"] is True
    assert w.final(coord, "j1").state == JobState.SUCCEEDED


def test_i1_a_machine_unknown_here_is_waited_for_until_its_deadline_and_grace(world):
    w = world(defaults={"deadline_margin": "1m"})
    w.pool.script("j1", FakeRun(polls=10_000), FakeRun())
    w.submit("j1", timeout="1m")
    coord = w.coordinator()
    job = running(w, coord, "j1")
    deadline = job.limits["deadline_at"]
    w.pool._machines.pop(job.machine)  # another host's pool, as far as this one knows

    keep_leases(w, coord, deadline - w.clock() + DEADLINE_GRACE_S - 20)
    found = run_audit(w, coord)
    assert "I1_running_without_machine" not in codes(found)
    assert w.job("j1").state == JobState.RUNNING

    keep_leases(w, coord, 30)
    found = run_audit(w, coord)
    assert codes(found) == ["I1_running_without_machine"]  # not also I6: lost first
    assert found[0].repaired
    assert [a.outcome for a in w.job("j1").attempts] == [Outcome.MACHINE_LOST]


def test_without_a_coordinator_job_repairs_are_only_reported(world):
    w = world()
    w.pool.script("j1", FakeRun(polls=500))
    w.submit("j1")
    coord = w.coordinator()
    job = running(w, coord, "j1")
    w.pool.lose(job.machine)

    found = run_audit(w)  # no coordinator: no claim token

    assert codes(found) == ["I1_running_without_machine"]
    assert not found[0].repaired
    assert w.job("j1").state == JobState.RUNNING and w.job("j1").attempts == []


# ---------------------------------------------------------------------- I2


def test_i2_a_remote_run_the_queue_does_not_place_there_is_cancelled(world):
    w = world()
    m = ready_machine(w)
    w.submit("j1")  # queued, never dispatched by anyone
    w.pool.dispatch(m, "j1-a1", w.tmp / "none.tar.gz", {}, {})

    found = run_audit(w)

    assert codes(found) == ["I2_duplicate_execution", "I7_stale_coordinator"]
    dup = found[0]
    assert dup.job_id == "j1" and dup.slot == m.slot and dup.repaired
    assert w.pool.cancels == ["j1-a1"]
    assert w.pool.remotes["j1-a1"].state == "cancelled"
    assert count(w, "I2_duplicate_execution") == 1


def test_i2_the_run_a_job_records_as_in_flight_is_not_a_duplicate(world):
    w = world()
    w.pool.script("j1", FakeRun(polls=500))
    w.submit("j1")
    coord = w.coordinator()
    running(w, coord, "j1")
    w.backend.release_coordinator(coord.worker_id)
    coord.leader = False

    assert "I2_duplicate_execution" not in codes(run_audit(w))
    assert w.pool.cancels == []


def test_machine_repairs_wait_while_another_coordinator_holds_the_lease(world):
    w = world()
    m = ready_machine(w)
    w.submit("j1")
    w.pool.dispatch(m, "j1-a1", w.tmp / "none.tar.gz", {}, {})
    w.pool.orphan_instances = [Instance(id="x9", provider="fake", label="jambu-p-s7-abcdef")]
    assert w.backend.acquire_coordinator("someone-else", 60)

    found = run_audit(w)

    assert sorted(codes(found)) == ["I2_duplicate_execution", "I4_orphan_machine"]
    assert not any(v.repaired for v in found)
    assert w.pool.cancels == [] and w.pool.retired_instances == []


# ---------------------------------------------------------------------- I3


def test_i3_a_success_whose_settle_never_landed_and_whose_outputs_broke_is_delivered_again(world, monkeypatch):
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
    ref = OutputRef(**w.job("j1").outputs_ref)
    (Path(ref.location) / "result.txt").write_bytes(b"damaged")
    w.clock.advance(120)  # every lease of the dead coordinator ran out

    second = w.coordinator("coord-b")
    job = w.final(second, "j1", max_ticks=40)

    [event] = reported(w, "I3_outputs_unverified")
    assert event["job_id"] == "j1" and event["repaired"] is True
    assert job.state == JobState.SUCCEEDED
    assert len(w.pool.starts) == 1
    assert [a.outcome for a in job.attempts] == [Outcome.SUCCEEDED]
    assert w.outputs.verify(OutputRef(**job.outputs_ref)) == []
    assert (Path(job.outputs_ref["location"]) / "result.txt").read_bytes() == b"done j1-a1"


def test_i3_a_settled_success_whose_outputs_broke_is_reported(world):
    w = world()
    w.submit("j1")
    coord = w.coordinator()
    job = w.final(coord, "j1")
    (Path(job.outputs_ref["location"]) / "result.txt").unlink()

    found = run_audit(w, coord)

    assert codes(found) == ["I3_outputs_unverified"]
    assert found[0].job_id == "j1" and not found[0].repaired
    assert "result.txt: missing" in found[0].detail
    assert count(w, "I3_outputs_unverified") == 1


# ---------------------------------------------------------------------- I4


def test_i4_an_orphan_machine_is_retired(world):
    w = world()
    w.pool.orphan_instances = [Instance(id="x9", provider="fake", label="jambu-p-s7-abcdef")]

    found = run_audit(w)

    assert codes(found) == ["I4_orphan_machine"]
    assert found[0].repaired and "x9" in found[0].detail
    assert w.pool.retired_instances and w.pool.retired_instances[0][0] == "x9"
    assert count(w, "I4_orphan_machine") == 1


def test_i4_an_orphan_that_is_a_job_s_machine_before_its_deadline_is_only_reported(world):
    w = world()
    w.submit("j1")
    now = w.clock()
    _ghost_in_flight(w, "j1", {
        "attempt": 1, "remote_id": "j1-a1", "slot": "s7", "instance_id": "x9",
        "started_at": now, "timeout_s": 600.0, "deadline_margin_s": 600.0, "deadline_at": now + 1200,
    })
    w.pool.orphan_instances = [Instance(id="x9", provider="fake", label="jambu-p-s7-abcdef")]

    found = [v for v in run_audit(w) if v.code == "I4_orphan_machine"]
    assert len(found) == 1 and not found[0].repaired and found[0].job_id == "j1"
    assert w.pool.retired_instances == []

    w.clock.now = now + 1200 + DEADLINE_GRACE_S + 1
    found = [v for v in run_audit(w) if v.code == "I4_orphan_machine"]
    assert len(found) == 1 and found[0].repaired
    assert [i for i, _ in w.pool.retired_instances] == ["x9"]


def test_the_coordinator_audits_every_twelve_ticks_and_repairs_machines(world):
    w = world()
    coord = w.coordinator()
    w.step(coord)  # leader; recover() audited an empty pool
    w.pool.orphan_instances = [Instance(id="x9", provider="fake", label="jambu-p-s7-abcdef")]

    reports = w.step(coord, 12)

    assert sum(r.violations for r in reports) == 1
    assert [i for i, _ in w.pool.retired_instances] == ["x9"]
    [event] = reported(w, "I4_orphan_machine")
    assert event["repaired"] is True


# ---------------------------------------------------------------------- I5


def test_i5_spend_over_the_cap_is_an_alert(world):
    w = world(max_usd_per_hour=1.5)
    ready_machine(w)
    ready_machine(w)  # 2 x $1/h

    found = run_audit(w)

    assert codes(found) == ["I5_spend_over_cap"]
    assert not found[0].repaired and found[0].job_id is None
    assert len(w.pool.machines()) == 2  # nothing is retired for it
    assert count(w, "I5_spend_over_cap") == 1


# ---------------------------------------------------------------------- I6


def test_i6_a_job_running_past_its_deadline_is_cancelled_and_requeued_timeout(world):
    w = world(defaults={"deadline_margin": "1m"})
    w.pool.script("j1", FakeRun(polls=10_000, outputs={"ckpt.bin": b"half"}), FakeRun())
    w.submit("j1", timeout="1m")
    coord = w.coordinator()
    job = running(w, coord, "j1")

    keep_leases(w, coord, job.limits["deadline_at"] - w.clock() + 5)
    found = run_audit(w, coord)

    assert codes(found) == ["I6_job_past_deadline"]
    assert found[0].repaired and found[0].job_id == "j1"
    assert w.pool.cancels == ["j1-a1"]
    job = w.job("j1")
    assert job.state == JobState.QUEUED
    assert [a.outcome for a in job.attempts] == [Outcome.TIMEOUT]
    assert w.outputs.verify(OutputRef(**job.outputs_ref)) == []  # what it wrote is kept
    assert count(w, "I6_job_past_deadline") == 1


def test_i6_without_a_coordinator_is_only_reported(world):
    w = world(defaults={"deadline_margin": "1m"})
    w.pool.script("j1", FakeRun(polls=10_000))
    w.submit("j1", timeout="1m")
    coord = w.coordinator()
    job = running(w, coord, "j1")
    keep_leases(w, coord, job.limits["deadline_at"] - w.clock() + 5)

    found = run_audit(w)

    assert codes(found) == ["I6_job_past_deadline"]
    assert not found[0].repaired and w.pool.cancels == []


# ---------------------------------------------------------------------- I7


def test_i7_no_live_coordinator_while_jobs_wait(world):
    w = world()
    w.submit("j1")

    found = run_audit(w)

    assert codes(found) == ["I7_stale_coordinator"]
    assert not found[0].repaired
    assert count(w, "I7_stale_coordinator") == 1


def test_i7_a_lease_holder_whose_heartbeat_is_older_than_three_ticks(world):
    w = world(tick="1s")
    assert w.backend.acquire_coordinator("slow-one", 15)
    w.clock.advance(2)
    assert run_audit(w) == []

    w.clock.advance(2)  # 4 s since its last renewal, 3 ticks are 3 s
    found = run_audit(w)
    assert codes(found) == ["I7_stale_coordinator"]
    assert "slow-one" in found[0].detail


# ---------------------------------------------------------------------- I8


def test_i8_fetched_outputs_not_stored_for_an_hour(world):
    w = world()
    work = Path(w.cfg.state_dir) / "queue" / "work" / "j1"
    (work / "out-a1").mkdir(parents=True)
    (work / "out-a1" / "model.bin").write_bytes(b"w")
    (work / "out-a2").mkdir()
    (work / "out-a2.stored").touch()
    old = w.clock() - 2 * 3600
    for p in (work / "out-a1", work / "out-a2"):
        os.utime(p, (old, old))

    found = run_audit(w)

    assert codes(found) == ["I8_outputs_not_stored"]
    assert found[0].job_id == "j1" and not found[0].repaired
    assert "attempt 1" in found[0].detail and "attempt 2" not in found[0].detail
    assert (work / "out-a1" / "model.bin").exists()  # alert only

    os.utime(work / "out-a1", (w.clock() - 60, w.clock() - 60))
    assert run_audit(w) == []


# ---------------------------------------------------------------------- reporting


def test_the_same_violation_is_reported_once_per_change_but_counted_every_time(world):
    w = world(max_usd_per_hour=0.5)
    ready_machine(w)
    memory = AuditMemory()

    for _ in range(3):
        assert codes(run_audit(w, memory=memory)) == ["I5_spend_over_cap"]

    assert len(reported(w, "I5_spend_over_cap")) == 1
    assert count(w, "I5_spend_over_cap") == 3

    w.pool._machines.clear()
    assert run_audit(w, memory=memory) == []
    ready_machine(w)
    run_audit(w, memory=memory)
    assert len(reported(w, "I5_spend_over_cap")) == 2  # it came back: a change
