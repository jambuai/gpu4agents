"""Engine: restarts, sessions, stopped_ttl, traffic as activity, session budget."""

from __future__ import annotations

import pytest
from fakes import FakeProvider

from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.engine import Engine
from gpu4agents.core.models import InstanceState, ProviderCapabilities
from gpu4agents.core.state import utc_now


def make_engine(config):
    provider = FakeProvider(config, CredentialResolver(config.project_dir))
    return Engine(config, provider=provider)


@pytest.fixture
def engine(config):
    return make_engine(config)


# -- reconcile vs. a stale watchdog `terminated` flag -----------------------


class _StaleWatchdog:
    """A watchdog that still remembers stopping the machine before a restart."""

    def try_state(self):
        return {"terminated": True, "locked": False, "starting": False}

    def events(self, limit: int = 100):
        return []


def test_reconcile_trusts_the_provider_over_a_stale_terminated_flag(engine, monkeypatch):
    state = engine.setup(wait=False)
    monkeypatch.setattr(engine, "beacon", lambda _state: _StaleWatchdog())

    reconciled = engine.reconcile()
    assert reconciled.state != "stopped"
    assert reconciled.running_since is not None  # still billing, so still guarded
    assert reconciled.stopped_since is None
    assert engine.provider.instances[state.instance_id].state is InstanceState.READY


def test_a_stale_terminated_flag_does_not_disable_the_guard(write_config, monkeypatch):
    engine = make_engine(write_config(lifecycle={"idle_timeout": "1m"}))
    engine.setup(wait=False)
    monkeypatch.setattr(engine, "beacon", lambda _state: _StaleWatchdog())
    with engine.store.transaction() as stored:
        stored.last_activity = utc_now() - 3600
        stored.starting = False

    decision = engine.guard_tick(apply=False)
    assert decision["action"] == "stop"
    assert decision["reason"] == "idle_timeout"


def test_terminated_only_counts_when_the_provider_cannot_tell(engine, monkeypatch):
    state = engine.setup(wait=False)
    monkeypatch.setattr(engine, "beacon", lambda _state: _StaleWatchdog())
    engine.provider.instances[state.instance_id].state = InstanceState.UNKNOWN

    assert engine.reconcile().state == "stopped"


def test_a_restart_outside_the_cli_is_seen_as_running_again(engine):
    state = engine.setup(wait=False)
    engine.stop()
    assert engine.store.load().state == "stopped"

    engine.provider.instances[state.instance_id].state = InstanceState.READY  # web console
    before = utc_now()
    reconciled = engine.reconcile()
    assert reconciled.state in ("ready", "idle")
    assert reconciled.stopped_since is None
    assert reconciled.session_started_at >= before


# -- sessions ---------------------------------------------------------------


def test_setup_starts_a_session(engine):
    state = engine.setup(wait=False)
    stored = engine.store.load()
    assert stored.session_started_at == pytest.approx(state.created_at, abs=1)
    assert stored.stopped_since is None


def test_stopping_records_stopped_since_and_destroy_clears_it(engine):
    engine.setup(wait=False)
    engine.stop()
    first = engine.store.load().stopped_since
    assert first is not None

    # A later reconcile keeps the first observation, not the latest.
    assert engine.reconcile().stopped_since == first

    engine.destroy()
    stored = engine.store.load()
    assert stored.stopped_since is None and stored.session_started_at is None


def test_a_stop_observed_by_reconcile_sets_stopped_since(engine):
    state = engine.setup(wait=False)
    engine.provider.instances[state.instance_id].state = InstanceState.STOPPED  # the watchdog did it
    assert engine.reconcile().stopped_since is not None


def test_restarting_a_stopped_instance_starts_a_new_session(engine, monkeypatch):
    engine.setup(wait=False)
    engine.stop()
    with engine.store.transaction() as stored:
        stored.session_started_at = utc_now() - 7 * 3600

    monkeypatch.setattr(engine, "_wait_running", lambda state: state.instance)
    monkeypatch.setattr(engine, "_connect_watchdog", lambda state: None)
    monkeypatch.setattr(engine, "_wait_runtime", lambda state: None)
    before = utc_now()
    state = engine.ensure_ready()

    assert ("start", state.instance_id) in engine.provider.calls
    assert state.session_started_at >= before
    assert state.stopped_since is None
    assert engine.store.load().session_started_at == pytest.approx(state.session_started_at)


def test_snapshot_carries_session_and_stop_times(engine):
    engine.setup(wait=False)
    engine.stop()
    stored = engine.store.load()
    snap = engine.snapshot(stored)
    assert snap.session_started_at == stored.session_started_at
    assert snap.stopped_since == stored.stopped_since


def test_a_restarted_instance_is_not_killed_for_its_age_before_the_restart(write_config):
    engine = make_engine(write_config(lifecycle={"max_lifetime": "6h", "idle_timeout": "15m"}))
    state = engine.setup(wait=False)
    engine.provider.instances[state.instance_id].created_at = utc_now() - 10 * 3600
    with engine.store.transaction() as stored:
        stored.session_started_at = utc_now() - 600  # restarted ten minutes ago
        stored.last_activity = utc_now()
        stored.starting = False

    assert engine.guard_tick()["action"] == "none"


def test_status_reports_session_fields(engine):
    engine.setup(wait=False)
    payload = engine.status()
    assert payload["session_started_at"] is not None
    assert payload["stopped_since"] is None


# -- stopped_ttl --------------------------------------------------------------


def test_policy_config_carries_stopped_ttl(write_config):
    assert make_engine(write_config()).policy_config().stopped_ttl_s == 24 * 3600
    disabled = make_engine(write_config(lifecycle={"stopped_ttl": None}))
    assert disabled.policy_config().stopped_ttl_s is None


def _stopped_hours_ago(engine, state, hours):
    """Backdate the whole history: ran, then stopped ``hours`` ago, nothing since."""
    ran = utc_now() - (hours + 1) * 3600
    engine.provider.instances[state.instance_id].created_at = ran
    with engine.store.transaction() as stored:
        stored.created_at = stored.session_started_at = stored.last_activity = ran
        stored.stopped_since = utc_now() - hours * 3600


def test_guard_destroys_an_instance_stopped_longer_than_stopped_ttl(write_config):
    engine = make_engine(write_config(lifecycle={"stopped_ttl": "1h"}))
    state = engine.setup(wait=False)
    engine.stop()
    _stopped_hours_ago(engine, state, 2)

    decision = engine.guard_tick()
    assert decision["action"] == "destroy"
    assert decision["reason"] == "stopped_ttl"
    assert ("destroy", state.instance_id) in engine.provider.calls
    assert engine.store.load().instance_id is None
    events = engine.events.tail(limit=50)
    triggered = [r for r in events if r["event"] == "lifecycle_guard_triggered"]
    destroyed = [r for r in events if r["event"] == "instance_destroyed"]
    assert triggered[-1]["reason"] == "stopped_ttl"
    assert destroyed[-1]["reason"] == "stopped_ttl"


def test_guard_leaves_a_recently_stopped_instance_alone(write_config):
    engine = make_engine(write_config(lifecycle={"stopped_ttl": "1h"}))
    engine.setup(wait=False)
    engine.stop()
    assert engine.guard_tick()["action"] == "none"
    assert engine.store.load().instance_id is not None


def test_dry_run_guard_reports_stopped_ttl_without_destroying(write_config):
    engine = make_engine(write_config(lifecycle={"stopped_ttl": "1h"}))
    state = engine.setup(wait=False)
    engine.stop()
    _stopped_hours_ago(engine, state, 2)
    assert engine.guard_tick(apply=False)["action"] == "destroy"
    assert not [c for c in engine.provider.calls if c[0] == "destroy"]


def test_a_stale_stopped_since_does_not_destroy_a_fresh_stop(write_config):
    """Bug: stopped_since kept the earliest sighting forever. Seen stopped two
    days ago, restarted from the console and idle-stopped by the watchdog
    minutes ago, the instance was destroyed as if it had sat stopped since."""
    engine = make_engine(write_config(lifecycle={"stopped_ttl": "24h"}))
    state = engine.setup(wait=False)
    engine.stop()
    with engine.store.transaction() as stored:
        stored.stopped_since = utc_now() - 48 * 3600  # it ran after this

    before = utc_now()
    decision = engine.guard_tick(apply=True)
    assert decision["action"] == "none"
    assert not [c for c in engine.provider.calls if c[0] == "destroy"]
    assert engine.store.load().stopped_since >= before - 1


def test_a_provider_start_time_after_stopped_since_restarts_the_stop_clock(write_config):
    engine = make_engine(write_config(lifecycle={"stopped_ttl": "24h"}))
    state = engine.setup(wait=False)
    engine.stop()
    _stopped_hours_ago(engine, state, 48)
    # The CLI never saw it run, but the provider says it was started 2h ago.
    engine.provider.instances[state.instance_id].created_at = utc_now() - 2 * 3600

    assert engine.guard_tick(apply=True)["action"] == "none"
    assert engine.store.load().stopped_since > utc_now() - 60


def test_without_evidence_of_running_the_first_stop_sighting_is_kept(write_config):
    engine = make_engine(write_config(lifecycle={"stopped_ttl": "24h"}))
    state = engine.setup(wait=False)
    engine.stop()
    _stopped_hours_ago(engine, state, 12)
    kept = engine.store.load().stopped_since
    assert engine.guard_tick(apply=True)["action"] == "none"
    assert engine.store.load().stopped_since == pytest.approx(kept, abs=1)


# -- the CLI guard and an unreachable watchdog -------------------------------


class _Unreachable:
    def try_state(self):
        return None

    def events(self, limit: int = 100):
        return []


def test_guard_leaves_idle_to_an_unreachable_watchdog(write_config, monkeypatch):
    """Bug: inference traffic is only counted on the instance. With the
    watchdog unreachable the CLI judged idleness from its own stale clock and
    stopped an endpoint that was busy serving requests."""
    engine = make_engine(write_config(lifecycle={"idle_timeout": "15m"}))
    state = engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.watchdog.installed = True
        stored.last_activity = utc_now() - 1200  # last CLI interaction 20 min ago
        stored.starting = False
    monkeypatch.setattr(engine, "beacon", lambda _state: _Unreachable())

    decision = engine.guard_tick(apply=True)
    assert decision["action"] == "none"
    assert decision["detail"]["idle_skipped"] == "watchdog_unreachable"
    assert ("stop", state.instance_id) not in engine.provider.calls
    assert engine.status()["decision"]["detail"]["idle_skipped"] == "watchdog_unreachable"


def test_max_lifetime_still_applies_when_the_watchdog_is_unreachable(write_config, monkeypatch):
    engine = make_engine(write_config(lifecycle={"idle_timeout": "15m", "max_lifetime": "6h"}))
    state = engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.watchdog.installed = True
        stored.session_started_at = utc_now() - 7 * 3600
        stored.starting = False
    monkeypatch.setattr(engine, "beacon", lambda _state: _Unreachable())

    decision = engine.guard_tick(apply=True)
    assert (decision["action"], decision["reason"]) == ("stop", "max_lifetime")
    assert ("stop", state.instance_id) in engine.provider.calls


def test_a_reachable_watchdog_lets_the_cli_apply_idle(write_config, monkeypatch):
    engine = make_engine(write_config(lifecycle={"idle_timeout": "15m"}))
    engine.setup(wait=False)
    old = utc_now() - 3600
    with engine.store.transaction() as stored:
        stored.last_activity = old
        stored.starting = False

    class Idle(_Unreachable):
        def try_state(self):
            return {"last_activity": old, "locked": False, "starting": False}

    monkeypatch.setattr(engine, "beacon", lambda _state: Idle())
    decision = engine.guard_tick(apply=False)
    assert (decision["action"], decision["reason"]) == ("stop", "idle_timeout")
    assert "idle_skipped" not in decision["detail"]


def test_a_stopped_instance_is_not_reported_as_idle_skipped(write_config, monkeypatch):
    """Seen live: `gpu status` on a stopped machine said idle was left to the watchdog."""
    engine = make_engine(write_config(lifecycle={"idle_timeout": "15m"}))
    state = engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.watchdog.installed = True
    engine.provider.stop(state.instance)
    monkeypatch.setattr(engine, "beacon", lambda _state: _Unreachable())

    decision = engine.guard_tick(apply=False)
    assert decision["reason"] == "already_terminal"
    assert "idle_skipped" not in decision["detail"]


def test_without_a_watchdog_the_cli_guard_owns_idle(write_config):
    engine = make_engine(
        write_config(lifecycle={"idle_timeout": "15m", "watchdog": {"enabled": False}})
    )
    engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.last_activity = utc_now() - 1200
        stored.starting = False
    assert engine.guard_tick(apply=False)["reason"] == "idle_timeout"


def test_validate_warns_when_nothing_removes_a_billing_stopped_instance(write_config):
    billing_stop = ProviderCapabilities(stop=True, destroy=True, stop_eliminates_billing=False)

    engine = make_engine(
        write_config(lifecycle={"stopped_ttl": None, "auto_destroy": False, "keep_disk": True})
    )
    engine.provider.capabilities = billing_stop
    result = engine.validate(check_provider=False)
    assert any(issue.field == "lifecycle.stopped_ttl" for issue in result.warnings)

    covered = make_engine(write_config(lifecycle={"stopped_ttl": "24h"}))
    covered.provider.capabilities = billing_stop
    result = covered.validate(check_provider=False)
    assert not any(issue.field == "lifecycle.stopped_ttl" for issue in result.warnings)

    # Without keep_disk the guard destroys, so no stopped instance is left.
    destroys = make_engine(write_config(lifecycle={"stopped_ttl": None}))
    destroys.provider.capabilities = billing_stop
    result = destroys.validate(check_provider=False)
    assert not any(issue.field == "lifecycle.stopped_ttl" for issue in result.warnings)

    # Where stopping ends billing, a kept stopped instance costs nothing.
    free_stop = make_engine(write_config(lifecycle={"stopped_ttl": None}))
    result = free_stop.validate(check_provider=False)
    assert not any(issue.field == "lifecycle.stopped_ttl" for issue in result.warnings)


# -- activity probe -----------------------------------------------------------


def _captured_agent_config(engine, monkeypatch):
    import gpu4agents.core.engine as engine_module

    captured = {}
    real = engine_module.build_agent_config

    def _capture(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(engine_module, "build_agent_config", _capture)
    engine._build_onstart("token")
    return captured


def test_the_watchdog_is_told_how_to_read_vllm_traffic(engine, monkeypatch):
    captured = _captured_agent_config(engine, monkeypatch)
    assert captured["activity_probe"] == {
        "kind": "prometheus",
        "url": "http://127.0.0.1:8000/metrics",
        "gauges": ["vllm:num_requests_running", "vllm:num_requests_waiting"],
        "counters": ["vllm:request_success_total", "vllm:prompt_tokens_total"],
    }


def test_traffic_can_be_excluded_from_activity(write_config, monkeypatch):
    engine = make_engine(write_config(lifecycle={"traffic_counts_as_activity": False}))
    assert _captured_agent_config(engine, monkeypatch)["activity_probe"] is None


def test_bare_compute_has_no_activity_probe(write_config, monkeypatch):
    engine = make_engine(write_config(runtime={"engine": "none"}))
    assert _captured_agent_config(engine, monkeypatch)["activity_probe"] is None


def test_the_watchdog_policy_carries_stopped_ttl(engine, monkeypatch):
    captured = _captured_agent_config(engine, monkeypatch)
    assert captured["policy"].stopped_ttl_s == 24 * 3600


# -- the guard's budget is per session and uses the hourly price -------------


def _budget_engine(write_config):
    engine = make_engine(
        write_config(
            lifecycle={"max_lifetime": "12h", "idle_timeout": "15m"},
            budget={"max_session_cost_usd": 2.0},
        )
    )
    engine.provider.offer_price = 1.0
    return engine


def test_guard_budget_uses_the_session_age_and_hourly_price(write_config):
    engine = _budget_engine(write_config)
    engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.session_started_at = utc_now() - 3 * 3600  # $3 at $1/h
        stored.last_activity = utc_now()
        stored.starting = False

    decision = engine.guard_tick(apply=False)
    assert decision["action"] == "stop"
    assert decision["reason"] == "budget_exceeded"
    assert decision["detail"]["session_cost_usd"] == pytest.approx(3.0, abs=0.01)
    assert engine.policy_config(1.0).hourly_cost_usd == 1.0


def test_guard_budget_resets_with_a_new_session(write_config):
    engine = _budget_engine(write_config)
    state = engine.setup(wait=False)
    engine.provider.instances[state.instance_id].created_at = utc_now() - 5 * 3600
    with engine.store.transaction() as stored:
        stored.session_started_at = utc_now() - 1800  # restarted: $0.50 so far
        stored.last_activity = utc_now()
        stored.starting = False

    assert engine.guard_tick(apply=False)["action"] == "none"


def test_a_state_file_from_before_sessions_counts_from_creation(engine):
    engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.session_started_at = None
        created = stored.created_at
    assert engine.reconcile().session_started_at == pytest.approx(created, abs=1)
