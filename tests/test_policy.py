"""The lifecycle rule is the whole point of the tool; test it hard."""

from __future__ import annotations

import pytest

from gpu4agents.core import policy as p

NOW = 10_000.0


def cfg(**overrides) -> p.PolicyConfig:
    base = dict(idle_timeout_s=900.0, max_lifetime_s=21_600.0, heartbeat_grace_s=300.0)
    base.update(overrides)
    return p.PolicyConfig(**base)


def snap(**overrides) -> p.Snapshot:
    base = dict(now=NOW, created_at=NOW - 60, last_activity=NOW)
    base.update(overrides)
    return p.Snapshot(**base)


def running(last_heartbeat: float, wid: str = "w1") -> p.Workload:
    return p.Workload(id=wid, started_at=NOW - 600, last_heartbeat=last_heartbeat)


def test_fresh_instance_is_left_alone():
    assert p.evaluate(snap(), cfg()).action == p.ACTION_NONE


def test_idle_beyond_timeout_stops():
    decision = p.evaluate(snap(last_activity=NOW - 1000), cfg())
    assert decision.action == p.ACTION_STOP
    assert decision.reason == p.REASON_IDLE


def test_idle_just_under_timeout_does_not_stop():
    assert p.evaluate(snap(last_activity=NOW - 899), cfg()).action == p.ACTION_NONE


def test_active_workload_is_never_stopped_by_the_idle_guard():
    # Criterion 5: last_activity is ancient, but a heartbeat is alive.
    decision = p.evaluate(
        snap(last_activity=NOW - 99_999, workloads=[running(NOW - 10)]), cfg()
    )
    assert decision.action == p.ACTION_NONE
    assert decision.reason == "workload_active"


def test_stale_heartbeat_falls_back_to_idle():
    # Criterion 9: the CLI died mid-run; after the grace period the guard acts.
    decision = p.evaluate(
        snap(last_activity=NOW - 5_000, workloads=[running(NOW - 4_000)]), cfg()
    )
    assert decision.action == p.ACTION_STOP
    assert decision.detail["stale_workloads"] == 1


def test_queued_workload_counts_as_activity():
    queued = p.Workload(id="w2", started_at=NOW, last_heartbeat=NOW - 9_999, state="queued")
    decision = p.evaluate(snap(last_activity=NOW - 9_999, workloads=[queued]), cfg())
    assert decision.action == p.ACTION_NONE


def test_startup_in_progress_is_activity():
    decision = p.evaluate(snap(last_activity=NOW - 9_999, starting=True), cfg())
    assert decision.reason == "startup_in_progress"


def test_max_lifetime_stops_even_when_busy():
    # Criterion 7: a forgotten instance cannot run forever just by staying busy.
    decision = p.evaluate(
        snap(created_at=NOW - 30_000, workloads=[running(NOW)]), cfg()
    )
    assert decision.action == p.ACTION_STOP
    assert decision.reason == p.REASON_MAX_LIFETIME


def test_lock_blocks_idle_shutdown():
    decision = p.evaluate(
        snap(last_activity=NOW - 9_999, locked=True, locked_until=NOW + 3_600), cfg()
    )
    assert decision.action == p.ACTION_NONE
    assert decision.reason == "locked"


def test_lock_blocks_max_lifetime_shutdown():
    decision = p.evaluate(
        snap(created_at=NOW - 99_999, locked=True, locked_until=NOW + 60), cfg()
    )
    assert decision.action == p.ACTION_NONE


def test_expired_lock_stops_protecting():
    # Criterion 8: after the TTL, normal rules resume.
    state = snap(last_activity=NOW - 9_999, locked=True, locked_until=NOW - 1)
    assert state.lock_expired(cfg()) is True
    assert p.evaluate(state, cfg()).action == p.ACTION_STOP


def test_indefinite_lock_is_ignored_unless_explicitly_allowed():
    state = snap(last_activity=NOW - 9_999, locked=True, locked_until=None)
    assert p.evaluate(state, cfg()).action == p.ACTION_STOP
    assert p.evaluate(state, cfg(allow_indefinite_lock=True)).action == p.ACTION_NONE


def test_auto_destroy_escalates_the_action():
    decision = p.evaluate(snap(last_activity=NOW - 9_999), cfg(auto_destroy=True))
    assert decision.action == p.ACTION_DESTROY


def test_destroy_when_the_provider_cannot_stop():
    decision = p.evaluate(snap(last_activity=NOW - 9_999), cfg(stop_supported=False))
    assert decision.action == p.ACTION_DESTROY


def test_no_action_when_both_switches_are_off():
    decision = p.evaluate(
        snap(last_activity=NOW - 9_999), cfg(auto_stop=False, auto_destroy=False)
    )
    assert decision.action == p.ACTION_NONE
    assert decision.reason == "idle_no_action"


def test_session_budget_ceiling_terminates():
    decision = p.evaluate(
        snap(created_at=NOW - 7_200), cfg(hourly_cost_usd=3.0, max_session_cost_usd=5.0)
    )
    assert decision.action == p.ACTION_STOP
    assert decision.reason == p.REASON_BUDGET


def test_no_timeouts_configured_means_no_shutdown():
    decision = p.evaluate(
        snap(last_activity=NOW - 999_999), cfg(idle_timeout_s=None, max_lifetime_s=None)
    )
    assert decision.action == p.ACTION_NONE


def test_terminal_states_are_left_alone():
    decision = p.evaluate(snap(provider_state=p.STOPPED), cfg())
    assert decision.reason == "already_terminal"


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, p.READY),
        ({"last_activity": NOW - 30}, p.IDLE),
        ({"starting": True}, p.STARTING),
        ({"workloads": [running(NOW)]}, p.BUSY),
        ({"provider_state": p.STOPPED}, p.STOPPED),
    ],
)
def test_derive_state(kwargs, expected):
    assert p.derive_state(snap(**kwargs), cfg()) == expected


# --- sessions: a restart is a new lease ------------------------------------


def test_session_age_falls_back_to_creation():
    assert snap(created_at=NOW - 500).session_age_seconds() == 500
    restarted = snap(created_at=NOW - 500, session_started_at=NOW - 20)
    assert restarted.session_age_seconds() == 20
    assert restarted.age_seconds() == 500


def test_max_lifetime_is_measured_from_the_session():
    # Created long ago, restarted a minute ago: must not be killed on boot.
    restarted = snap(created_at=NOW - 30_000, session_started_at=NOW - 60, workloads=[running(NOW)])
    assert p.evaluate(restarted, cfg()).action == p.ACTION_NONE
    forgotten = snap(created_at=NOW - 90_000, session_started_at=NOW - 30_000)
    decision = p.evaluate(forgotten, cfg())
    assert decision.reason == p.REASON_MAX_LIFETIME
    assert decision.detail["session_age_seconds"] == 30_000


def test_budget_is_spent_per_session():
    budget = cfg(hourly_cost_usd=3.0, max_session_cost_usd=5.0)
    fresh = snap(created_at=NOW - 7_200, session_started_at=NOW - 600)
    assert p.evaluate(fresh, budget).action == p.ACTION_NONE
    spent = snap(created_at=NOW - 99_999, session_started_at=NOW - 7_200)
    decision = p.evaluate(spent, budget)
    assert decision.reason == p.REASON_BUDGET
    assert decision.detail["session_cost_usd"] == 6.0


# --- stopped instances still bill storage ----------------------------------


def stopped(**overrides) -> p.Snapshot:
    base = dict(provider_state=p.STOPPED, stopped_since=NOW - 7_200)
    base.update(overrides)
    return snap(**base)


def test_a_long_stopped_instance_is_destroyed():
    decision = p.evaluate(stopped(), cfg(stopped_ttl_s=3_600.0))
    assert decision.action == p.ACTION_DESTROY
    assert decision.reason == p.REASON_STOPPED_TTL == "stopped_ttl"
    assert decision.detail["stopped_seconds"] == 7_200


@pytest.mark.parametrize(
    "state,config",
    [
        (stopped(), cfg()),  # no TTL configured
        (stopped(stopped_since=None), cfg(stopped_ttl_s=3_600.0)),  # unknown since when
        (stopped(stopped_since=NOW - 3_000), cfg(stopped_ttl_s=3_600.0)),  # under the TTL
        (stopped(), cfg(stopped_ttl_s=3_600.0, destroy_supported=False)),
        (stopped(locked=True, locked_until=NOW + 60), cfg(stopped_ttl_s=3_600.0)),
        (snap(provider_state=p.DESTROYED, stopped_since=NOW - 7_200), cfg(stopped_ttl_s=1.0)),
    ],
)
def test_stopped_ttl_negatives_stay_terminal(state, config):
    decision = p.evaluate(state, config)
    assert decision.action == p.ACTION_NONE
    assert decision.reason == "already_terminal"


def test_stopped_ttl_only_applies_to_stopped_instances():
    live = snap(stopped_since=NOW - 99_999)
    assert p.evaluate(live, cfg(stopped_ttl_s=1.0)).action == p.ACTION_NONE


def test_policy_config_from_dict_ignores_unknown_keys():
    config = p.PolicyConfig.from_dict({"stopped_ttl_s": 60.0, "from_the_future": 1})
    assert config.stopped_ttl_s == 60.0
    assert p.PolicyConfig().stopped_ttl_s is None
    assert "stopped_ttl_s" in config.to_dict()
