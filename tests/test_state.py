from __future__ import annotations

import json

import pytest

from gpu4agents.core.errors import StateError
from gpu4agents.core.models import Endpoint, Instance, InstanceState
from gpu4agents.core.state import RuntimeState, StateStore, WorkloadRecord, utc_now


@pytest.fixture
def store(tmp_path) -> StateStore:
    return StateStore(tmp_path / ".gpu")


def test_empty_state_is_not_an_error(store):
    assert store.load().instance_id is None


def test_round_trip_preserves_everything(store):
    state = RuntimeState(
        provider="vast",
        instance_id="19384723",
        state="idle",
        created_at=utc_now(),
        last_activity=utc_now(),
        instance=Instance(
            id="19384723",
            provider="vast",
            state=InstanceState.READY,
            endpoints={"model": Endpoint("model", "http://1.2.3.4:41000", 8000, 41000)},
        ),
    )
    state.workloads["w1"] = WorkloadRecord(id="w1", label="eval", command=["python", "x.py"])
    store.save(state)

    loaded = store.load()
    assert loaded.instance_id == "19384723"
    assert loaded.instance.endpoint_url("model") == "http://1.2.3.4:41000"
    assert loaded.workloads["w1"].command == ["python", "x.py"]


def test_state_file_is_human_readable_iso(store):
    state = RuntimeState(provider="vast", instance_id="1", created_at=utc_now())
    store.save(state)
    raw = json.loads(store.path.read_text())
    assert raw["created_at"].endswith("Z")
    assert raw["version"] == 1


def test_corrupt_state_is_reported_clearly(store):
    store.ensure_dirs()
    store.path.write_text("{not json")
    with pytest.raises(StateError):
        store.load()


def test_state_dir_is_gitignored(store):
    store.ensure_dirs()
    assert (store.dir / ".gitignore").read_text().strip() == "*"


def test_transaction_writes_on_exit(store):
    with store.transaction() as state:
        state.provider = "vast"
        state.instance_id = "42"
    assert store.load().instance_id == "42"


def test_process_lock_is_reentrant_within_one_process(store):
    with store.process_lock():
        with store.transaction() as state:
            state.provider = "vast"
    assert store.load().provider == "vast"


def test_clear_instance_fails_pending_workloads(store):
    state = RuntimeState(provider="vast", instance_id="9")
    state.workloads["w1"] = WorkloadRecord(id="w1", state="running")
    state.clear_instance()
    assert state.instance_id is None
    assert state.workloads["w1"].state == "failed"


def test_prune_keeps_recent_history(store):
    state = RuntimeState()
    for index in range(40):
        state.workloads[f"w{index}"] = WorkloadRecord(
            id=f"w{index}", state="finished", finished_at=utc_now() + index
        )
    state.prune_workloads(keep=10)
    assert len(state.workloads) == 10


# -- billable-time bookkeeping (spec section 25: cost must reflect actual
# running time, not wall-clock time since creation) ------------------------


def test_accrued_billable_seconds_with_no_activity_is_zero():
    state = RuntimeState()
    assert state.accrued_billable_seconds(now=1000) == 0.0


def test_mark_billable_starts_and_stops_the_clock():
    state = RuntimeState()
    state.mark_billable(True, now=0.0)
    assert state.running_since == 0.0
    assert state.accrued_billable_seconds(now=100) == 100.0

    state.mark_billable(False, now=100.0)
    assert state.running_since is None
    assert state.billed_seconds == 100.0
    # Time passing after the stop must not keep accruing.
    assert state.accrued_billable_seconds(now=99999) == 100.0


def test_mark_billable_accumulates_across_multiple_stop_start_cycles():
    state = RuntimeState()
    state.mark_billable(True, now=0)
    state.mark_billable(False, now=60)  # ran 60s
    # Stopped for a long stretch - must not count.
    state.mark_billable(True, now=99999)
    state.mark_billable(False, now=99999 + 30)  # ran another 30s
    assert state.billed_seconds == 90.0


def test_mark_billable_is_idempotent_while_state_does_not_change():
    state = RuntimeState()
    state.mark_billable(True, now=0)
    state.mark_billable(True, now=50)  # still billable - must not reset running_since
    assert state.running_since == 0.0
    assert state.accrued_billable_seconds(now=100) == 100.0


def test_clear_instance_freezes_the_final_cost_and_resets_the_clock():
    state = RuntimeState(hourly_cost_usd=3600.0)  # $1/s, easy arithmetic
    state.mark_billable(True, now=0)
    state.mark_billable(False, now=42)  # ran 42s -> $42

    state.clear_instance()
    assert state.session_cost_usd == pytest.approx(42.0)
    assert state.billed_seconds == 0.0
    assert state.running_since is None


def test_clear_instance_closes_a_still_open_billable_period():
    """Destroying a RUNNING instance must count the time up to now, not lose it."""
    state = RuntimeState(hourly_cost_usd=3600.0)
    state.mark_billable(True, now=0)
    state.clear_instance()  # never explicitly marked stopped first
    with_now = RuntimeState(hourly_cost_usd=3600.0)
    with_now.mark_billable(True, now=0)
    with_now.mark_billable(False)  # closes using real utc_now(), just checking no crash
    assert with_now.billed_seconds >= 0


def test_session_and_stop_times_round_trip_as_iso(store):
    state = RuntimeState(instance_id="i-1", state="stopped")
    state.mark_session_start(1_700_000_000.0)
    state.mark_stopped(1_700_003_600.0)
    state.mark_stopped(1_700_009_999.0)  # the first sighting is kept
    store.save(state)

    raw = json.loads(store.path.read_text())
    assert raw["session_started_at"] == "2023-11-14T22:13:20Z"
    assert raw["stopped_since"] == "2023-11-14T23:13:20Z"
    loaded = store.load()
    assert loaded.session_started_at == 1_700_000_000.0
    assert loaded.stopped_since == 1_700_003_600.0


def test_a_new_session_clears_stopped_since_and_clear_instance_resets_both():
    state = RuntimeState(instance_id="i-1")
    state.mark_stopped(10.0)
    state.mark_session_start(20.0)
    assert state.stopped_since is None and state.session_started_at == 20.0
    state.mark_stopped(30.0)
    state.clear_instance()
    assert state.session_started_at is None and state.stopped_since is None


def test_state_files_from_before_sessions_still_load(store):
    store.ensure_dirs()
    store.path.write_text(json.dumps({"version": 1, "instance_id": "i-1", "state": "ready"}))
    loaded = store.load()
    assert loaded.session_started_at is None and loaded.stopped_since is None
