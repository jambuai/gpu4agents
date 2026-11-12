import pytest

from gpu4agents.queue.pool_types import MachineState, PoolMachine
from gpu4agents.queue.spend import SpendLedger, pool_usd_per_hour


def _pm(slot, state, hourly=0.5, stopped=0.05, jobs=None):
    return PoolMachine(
        slot=slot, instance_id=f"i-{slot}", state=state, hourly_usd=hourly, stopped_hourly_usd=stopped,
        ready_at=None, deadline_at=None, jobs=jobs or [], health_failures=0, last_seen_at=None, opened_at=0.0,
    )


def test_stopped_machines_count_their_disk():
    ms = [_pm("a", MachineState.BUSY), _pm("b", MachineState.STOPPED), _pm("c", MachineState.GONE)]
    assert pool_usd_per_hour(ms) == pytest.approx(0.55)


def test_held_machines_count_full_rate():
    assert pool_usd_per_hour([_pm("a", MachineState.HELD), _pm("b", MachineState.RETIRED)]) == pytest.approx(0.5)


def test_job_cost_splits_a_shared_machine(tmp_path):
    led = SpendLedger(tmp_path / "spend.jsonl")
    assert led.record("a", 0, 3600, 1.0, jobs=["j1", "j2"]) == pytest.approx(1.0)
    led.record("a", 3600, 5400, 1.0, jobs=["j1"])
    led.record("a", 5400, 7200, 1.0)  # idle: in the total, in no job
    assert led.by_job() == {"j1": pytest.approx(1.0), "j2": pytest.approx(0.5)}
    assert led.total() == pytest.approx(2.0)


def test_ledger_survives_restart(tmp_path):
    p = tmp_path / "q" / "spend.jsonl"
    SpendLedger(p).record("a", 0, 1800, 2.0, jobs=["j1"])
    again = SpendLedger(p)
    assert again.total() == pytest.approx(1.0)
    assert again.by_job() == {"j1": pytest.approx(1.0)}


def test_a_torn_last_line_does_not_swallow_the_next_row(tmp_path):
    p = tmp_path / "spend.jsonl"
    led = SpendLedger(p)
    led.record("a", 0, 3600, 1.0, jobs=["j1"])
    led.record("a", 0, 3600, 1.0, jobs=["j2"])
    data = p.read_bytes()
    p.write_bytes(data[:-10])  # crash mid-line
    again = SpendLedger(p)
    assert again.total() == pytest.approx(1.0)
    again.record("a", 0, 3600, 2.0, jobs=["j3"])
    fresh = SpendLedger(p)
    assert fresh.total() == pytest.approx(3.0)
    assert fresh.by_job() == {"j1": pytest.approx(1.0), "j3": pytest.approx(2.0)}


def test_malformed_rows_are_skipped_and_totals_stay_in_memory(tmp_path):
    p = tmp_path / "spend.jsonl"
    p.write_text('[1]\n"x"\n{"usd": "a"}\n{}\n{"usd": 1.5, "jobs": ["j"]}\n')
    led = SpendLedger(p)
    assert led.total() == pytest.approx(1.5)
    led.record("a", 0, 3600, 1.0, jobs=["j"])
    p.unlink()
    assert led.total() == pytest.approx(2.5)
    assert led.by_job() == {"j": pytest.approx(2.5)}
