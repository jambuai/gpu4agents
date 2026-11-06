"""Orchestration: idempotency, reconciliation, cleanup, budget, the guard."""

from __future__ import annotations

import pytest
from fakes import FakeProvider

from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.engine import Engine
from gpu4agents.core.errors import BudgetExceededError, ConfigError, ProvisioningError
from gpu4agents.core.models import InstanceState
from gpu4agents.core.state import utc_now


@pytest.fixture
def engine(config):
    provider = FakeProvider(config, CredentialResolver(config.project_dir))
    return Engine(config, provider=provider)


def make_engine(config):
    provider = FakeProvider(config, CredentialResolver(config.project_dir))
    return Engine(config, provider=provider)


# Without a watchdog the CLI guard owns the idle rule; with one it defers to it
# (see test_engine_sessions.py::test_guard_leaves_idle_to_an_unreachable_watchdog).
LOCAL_IDLE = {"idle_timeout": "1m", "watchdog": {"enabled": False}}


# -- idempotency (spec section 23) ------------------------------------------


def test_setup_twice_does_not_create_two_instances(engine):
    first = engine.setup(wait=False)
    second = engine.setup(wait=False)
    assert first.instance_id == second.instance_id
    assert engine.provider.counter == 1


def test_setup_refuses_to_reuse_an_incompatible_instance(write_config):
    config = write_config()
    engine = make_engine(config)
    engine.setup(wait=False)

    changed = write_config(model={"id": "org/some-other-model"})
    other = make_engine(changed)
    other.provider.instances = engine.provider.instances
    with pytest.raises(ConfigError) as exc:
        other.setup(wait=False)
    assert "--recreate" in str(exc.value)


def test_recreate_replaces_the_instance(write_config):
    config = write_config()
    engine = make_engine(config)
    first = engine.setup(wait=False)
    second = engine.setup(wait=False, recreate=True)
    assert second.instance_id != first.instance_id
    assert ("destroy", first.instance_id) in engine.provider.calls


def test_stop_on_an_already_stopped_instance_succeeds(engine):
    engine.setup(wait=False)
    engine.stop()
    result = engine.stop()
    assert result["changed"] is False


def test_destroy_with_nothing_provisioned_succeeds(engine):
    assert engine.destroy()["changed"] is False


# -- reconciliation (spec section 18) ---------------------------------------


def test_status_repairs_state_when_the_instance_vanished(engine):
    state = engine.setup(wait=False)
    engine.provider.instances.clear()  # someone destroyed it in the web console

    payload = engine.status()
    assert payload["state"] == "none"
    assert engine.store.load().instance_id is None
    assert any(
        record["event"] == "state_reconciled" for record in engine.events.tail(limit=50)
    )


def test_reconciliation_marks_orphaned_workloads_failed(engine):
    state = engine.setup(wait=False)
    with engine.store.transaction() as stored:
        from gpu4agents.core.state import WorkloadRecord

        stored.workloads["w1"] = WorkloadRecord(id="w1", state="running")
    engine.provider.stop(engine.provider.instances[state.instance_id])

    engine.reconcile()
    assert engine.store.load().workloads["w1"].state == "failed"


# -- failure cleanup (spec criterion 4) --------------------------------------


def test_failed_setup_cleans_up_partially_created_resources(engine):
    engine.provider.fail_setup = ProvisioningError("boom")
    with pytest.raises(ProvisioningError):
        engine.setup(wait=False)

    events = [record["event"] for record in engine.events.tail(limit=50)]
    assert "setup_failed" in events
    assert engine.store.load().instance_id is None


def test_an_interrupted_setup_does_not_leave_the_machine_billing(engine, monkeypatch):
    """Caught live: Ctrl-C during the image pull left the instance booting."""

    def interrupted(state=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "ensure_ready", interrupted)
    with pytest.raises(KeyboardInterrupt):
        engine.setup()

    assert engine.provider.instances == {}
    assert engine.store.load().instance_id is None


def test_cleanup_can_be_disabled(write_config):
    config = write_config(lifecycle={"cleanup_on_setup_failure": False})
    engine = make_engine(config)
    engine.provider.fail_setup = ProvisioningError("boom")
    with pytest.raises(ProvisioningError):
        engine.setup(wait=False)
    assert "cleanup_performed" not in [r["event"] for r in engine.events.tail(limit=50)]


# -- budget (spec section 17) ------------------------------------------------


def test_provisioning_above_the_hourly_budget_fails(write_config):
    config = write_config(budget={"max_hourly_cost_usd": 0.10})
    engine = make_engine(config)
    engine.provider.offer_price = 0.90
    with pytest.raises(BudgetExceededError) as exc:
        engine.setup(wait=False)
    assert "--allow-cost-override" in str(exc.value)
    assert engine.provider.counter == 0


def test_cost_override_provisions_anyway(write_config):
    config = write_config(budget={"max_hourly_cost_usd": 0.10})
    engine = make_engine(config)
    engine.provider.offer_price = 0.90
    state = engine.setup(wait=False, allow_cost_override=True)
    assert state.instance_id is not None


# -- guard (spec sections 10, 14) -------------------------------------------


def test_guard_stops_an_idle_unlocked_instance_and_records_the_reason(write_config):
    config = write_config(lifecycle=LOCAL_IDLE)
    engine = make_engine(config)
    state = engine.setup(wait=False)

    with engine.store.transaction() as stored:
        stored.last_activity = utc_now() - 3600
        stored.starting = False

    decision = engine.guard_tick()
    assert decision["action"] == "stop"
    assert decision["reason"] == "idle_timeout"

    guard_events = [
        r for r in engine.events.tail(limit=50) if r["event"] == "lifecycle_guard_triggered"
    ]
    assert guard_events and guard_events[-1]["reason"] == "idle_timeout"
    assert ("stop", state.instance_id) in engine.provider.calls


def test_guard_respects_an_active_lock(write_config):
    config = write_config(lifecycle=LOCAL_IDLE)
    engine = make_engine(config)
    engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.last_activity = utc_now() - 3600
        stored.starting = False
    engine.lock(3600, reason="benchmark running")

    assert engine.guard_tick()["action"] == "none"


def test_expired_lock_stops_protecting(write_config):
    config = write_config(lifecycle=LOCAL_IDLE)
    engine = make_engine(config)
    engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.last_activity = utc_now() - 3600
        stored.starting = False
        stored.locked = True
        stored.locked_until = utc_now() - 1

    assert engine.guard_tick()["action"] == "stop"


def test_indefinite_lock_is_refused_by_default(engine):
    engine.setup(wait=False)
    with pytest.raises(ConfigError):
        engine.lock(None)


def test_dry_run_guard_changes_nothing(write_config):
    config = write_config(lifecycle=LOCAL_IDLE)
    engine = make_engine(config)
    engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.last_activity = utc_now() - 3600
        stored.starting = False

    assert engine.guard_tick(apply=False)["action"] == "stop"
    assert not [c for c in engine.provider.calls if c[0] == "stop"]


# -- validation --------------------------------------------------------------


def test_validation_flags_an_unbounded_lifecycle(write_config):
    config = write_config(lifecycle={"idle_timeout": None, "max_lifetime": None})
    result = make_engine(config).validate()
    assert not result.ok
    assert any("idle_timeout" in issue.message for issue in result.errors)


def test_validation_warns_when_the_watchdog_is_disabled(write_config):
    config = write_config(lifecycle={"watchdog": {"enabled": False}})
    result = make_engine(config).validate()
    assert any("watchdog" in (issue.field or "") for issue in result.warnings)


# -- workload environment ----------------------------------------------------


def test_workload_env_points_at_the_remote_runtime(engine):
    state = engine.setup(wait=False)
    state.endpoint = "http://127.0.0.1:9"
    env = engine.workload_env(state, "w1")
    assert env["JAMBU_GPU_ENDPOINT"] == "http://127.0.0.1:9"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9/v1"
    assert env["JAMBU_MODEL_ID"] == "empero-ai/Qwythos-9B-v2"
    assert env["JAMBU_WORKLOAD_ID"] == "w1"


def test_setup_locks_the_endpoint_with_a_generated_key(engine):
    state = engine.setup(wait=False)
    assert state.api_key and len(state.api_key) >= 40
    assert f"export VLLM_API_KEY='{state.api_key}'" in engine.provider.last_spec.onstart
    state.endpoint = "http://127.0.0.1:9"
    env = engine.workload_env(state, "w1")
    assert env["OPENAI_API_KEY"] == state.api_key


def test_each_instance_gets_its_own_key(engine):
    first = engine.setup(wait=False).api_key
    second = engine.setup(wait=False, recreate=True).api_key
    assert first != second


def test_a_configured_key_is_used_and_not_repeated_in_the_environment(write_config):
    engine = make_engine(write_config(runtime={"api_key": "my-key"}))
    state = engine.setup(wait=False)
    assert state.api_key == "my-key"
    assert "VLLM_API_KEY" not in engine.provider.last_spec.onstart
    # The server's arguments travel deflated inside the boot script.
    assert "--api-key my-key" in engine.runtime.launcher_script()
    assert engine.runtime.start_command() in engine.provider.last_spec.onstart


def test_an_instance_from_before_generated_keys_still_gets_a_client_key(engine):
    state = engine.setup(wait=False)
    state.api_key = None
    state.endpoint = "http://127.0.0.1:9"
    assert engine.workload_env(state, "w1")["OPENAI_API_KEY"] == "jambu-local"


def test_connection_is_what_a_client_needs(write_config):
    engine = make_engine(write_config(model={"id": "org/m", "served_name": "m"}))
    state = engine.setup(wait=False)
    state.endpoint = "http://1.2.3.4:5/"
    conn = engine.connection(state)
    assert conn["base_url"] == "http://1.2.3.4:5/v1"
    assert conn["api_key"] == state.api_key
    assert conn["model"] == "m"
    assert engine.workload_env(state, "w1")["JAMBU_MODEL_ID"] == "m"


def test_connection_without_an_instance_has_no_key(engine):
    from gpu4agents.core.state import RuntimeState

    conn = engine.connection(RuntimeState())
    assert conn["state"] == "none"
    assert conn["endpoint"] is None and conn["api_key"] is None


# -- provider isolation (spec criterion 2, 11) -------------------------------


def test_the_core_never_branches_on_a_provider_name():
    import pathlib

    core = pathlib.Path(__file__).resolve().parents[1] / "gpu4agents" / "core"
    offenders = []
    for path in core.rglob("*.py"):
        text = path.read_text()
        for needle in ('== "vast"', "== 'vast'", 'name == "vast"'):
            if needle in text:
                offenders.append(str(path))
    assert not offenders, f"core must not branch on provider names: {offenders}"


def test_a_new_provider_needs_only_the_contract(config):
    """The engine drives the fake provider end to end with zero core changes."""
    engine = make_engine(config)
    state = engine.setup(wait=False)
    assert state.provider == "fake"
    assert engine.status()["provider"] == "fake"
    assert engine.logs() == ["line one", "line two"]
    engine.destroy()
    assert engine.store.load().instance_id is None


# -- run flow (spec section 16) ---------------------------------------------


def test_run_executes_the_command_and_records_the_workload(engine, monkeypatch):
    state = engine.setup(wait=False)
    state.endpoint = "http://127.0.0.1:9"
    monkeypatch.setattr(engine, "ensure_ready", lambda: state)

    import sys

    assert engine.run([sys.executable, "-c", "print('hello')"], label="smoke") == 0

    stored = engine.store.load()
    record = list(stored.workloads.values())[-1]
    assert record.state == "finished"
    assert record.returncode == 0
    assert record.label == "smoke"

    events = [r["event"] for r in engine.events.tail(limit=50)]
    assert "workload_started" in events and "workload_finished" in events


def test_a_failing_workload_propagates_its_exit_code(engine, monkeypatch):
    from gpu4agents.core.errors import WorkloadFailed

    state = engine.setup(wait=False)
    monkeypatch.setattr(engine, "ensure_ready", lambda: state)

    import sys

    with pytest.raises(WorkloadFailed) as exc:
        engine.run([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert exc.value.returncode == 3
    assert list(engine.store.load().workloads.values())[-1].state == "failed"


def test_the_instance_is_not_stopped_after_a_workload(engine, monkeypatch):
    """Sequential experiments reuse the instance; idle_timeout governs teardown."""
    state = engine.setup(wait=False)
    monkeypatch.setattr(engine, "ensure_ready", lambda: state)

    import sys

    engine.run([sys.executable, "-c", "pass"])
    assert engine.store.load().instance_id == state.instance_id
    assert not [call for call in engine.provider.calls if call[0] in ("stop", "destroy")]


def test_run_marks_activity_so_the_guard_sees_it(engine, monkeypatch):
    config_state = engine.setup(wait=False)
    monkeypatch.setattr(engine, "ensure_ready", lambda: config_state)

    import sys

    before = engine.store.load().last_activity
    engine.run([sys.executable, "-c", "pass"])
    assert engine.store.load().last_activity >= before


# -- billable-time bookkeeping (found live: `status` checked long after an
# instance auto-stopped kept reporting cost as if it billed the whole
# wall-clock gap, not just the time it was actually running) ----------------


def test_session_cost_stops_growing_once_the_instance_is_stopped(engine):
    """The instance actually stopped 15 simulated minutes after creation
    (running_since already cleared, billed_seconds closed at that point) -
    checking status hours of wall-clock time later must not add a cent.
    """
    state = engine.setup(wait=False)
    engine.provider.instances[state.instance_id].state = InstanceState.STOPPED
    with engine.store.transaction() as stored:
        stored.state = "stopped"
        stored.running_since = None  # already closed by an earlier reconcile
        stored.billed_seconds = 900.0  # ran for 15 real minutes before stopping
        stored.created_at = utc_now() - 99999  # then hours passed before anyone checked

    payload = engine.status()
    expected = engine.provider.offer_price * 900.0 / 3600.0
    assert payload["session_cost_usd"] == pytest.approx(expected, abs=0.01)


def test_stopping_a_running_instance_freezes_the_cost_at_the_stop_point(write_config):
    config = write_config()
    engine = make_engine(config)
    state = engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.created_at = utc_now() - 10
        stored.running_since = stored.created_at  # ran for ~10s

    engine.stop()
    cost_at_stop = engine.store.load().session_cost_usd
    expected = engine.provider.offer_price * 10.0 / 3600.0
    assert cost_at_stop == pytest.approx(expected, abs=0.01)
    assert cost_at_stop > 0

    # Checking status again much later must not inflate it further.
    with engine.store.transaction() as stored:
        stored.created_at = utc_now() - 99999  # simulate a long time passing
    payload = engine.status()
    assert payload["session_cost_usd"] == pytest.approx(cost_at_stop, abs=0.01)


def test_setup_starts_the_billable_clock_immediately_not_at_the_next_reconcile(engine):
    state = engine.setup(wait=False)
    assert engine.store.load().running_since is not None


# -- the runtime image sets the CUDA floor ------------------------------------


def _engine(write_config, **overrides):
    config = write_config(**overrides)
    return Engine(config, provider=FakeProvider(config, CredentialResolver(config.project_dir)))


def test_the_spec_asks_for_the_cuda_the_image_needs(write_config):
    assert _engine(write_config).build_spec().gpu.cuda_min == "13.0"
    older = _engine(write_config, compute={"image": "vllm/vllm-openai:v0.29.0-cu129"})
    assert older.build_spec().gpu.cuda_min == "12.9"


def test_a_stricter_configured_cuda_floor_wins_and_a_looser_one_loses(write_config):
    strict = _engine(write_config, compute={"gpu": {"cuda_min": "13.1"}})
    assert strict.build_spec().gpu.cuda_min == "13.1"
    loose = _engine(write_config, compute={"gpu": {"cuda_min": "12.1"}})
    assert loose.build_spec().gpu.cuda_min == "13.0"


def test_an_image_of_unknown_cuda_keeps_the_configured_floor(write_config):
    custom = {"image": "someone/server:v2"}
    assert _engine(write_config, compute=custom).build_spec().gpu.cuda_min is None
    pinned = dict(custom, gpu={"cuda_min": "12.4"})
    assert _engine(write_config, compute=pinned).build_spec().gpu.cuda_min == "12.4"


def test_the_spec_floors_the_gpu_architecture_the_runtime_needs(write_config):
    # Default CUDA-13 vLLM image -> Ampere (8.0).
    assert _engine(write_config).build_spec().gpu.compute_cap_min == 8.0


def test_a_stricter_configured_compute_cap_wins_and_a_looser_one_loses(write_config):
    strict = _engine(write_config, compute={"gpu": {"compute_cap_min": 8.9}})
    assert strict.build_spec().gpu.compute_cap_min == 8.9  # e.g. force Ada
    loose = _engine(write_config, compute={"gpu": {"compute_cap_min": 7.0}})
    assert loose.build_spec().gpu.compute_cap_min == 8.0  # runtime floor still wins


def test_validate_warns_about_an_unpinned_image(write_config):
    for image in ("vllm/vllm-openai:latest", "vllm/vllm-openai"):
        result = _engine(write_config, compute={"image": image}).validate(check_provider=False)
        assert any("unpinned image" in issue.message for issue in result.warnings), image
    pinned = _engine(write_config).validate(check_provider=False)
    assert not any("unpinned image" in issue.message for issue in pinned.warnings)


def test_validate_previews_offers_with_the_image_cuda_floor(write_config):
    engine = _engine(write_config)
    seen = {}

    def _validate(config):
        seen["cuda_min"] = config.compute.gpu.cuda_min
        from gpu4agents.core.models import ValidationResult

        return ValidationResult()

    engine.provider.validate = _validate
    engine.validate()
    assert seen["cuda_min"] == "13.0"
    assert engine.config.compute.gpu.cuda_min is None  # the loaded config is untouched


def test_the_watchdog_learns_the_runtimes_pid_and_log(engine, monkeypatch):
    import json
    import re
    import base64
    import zlib

    script = engine._build_onstart("t" * 43)
    blob = re.search(r"echo '(\S+)' > /etc/jambu/watchdog.json.b85", script).group(1)
    config = json.loads(zlib.decompress(base64.b85decode(blob)))
    assert config["runtime_pid_path"] == "/var/run/jambu/vllm.pid"
    assert config["logs"] == {
        "runtime": "/var/log/jambu/vllm.log",
        "onstart": "/var/log/jambu/onstart.log",
        "watchdog": "/var/log/jambu/watchdog.log",
    }


# -- a model server that died at boot fails setup fast, with its log ----------


class _DeadRuntimeWatchdog:
    """A reachable watchdog reporting that the model server process is gone."""

    def __init__(self, alive=False):
        self.alive = alive
        self.log_requests = []

    def ping(self):
        return True

    def identify(self, *_args, **_kwargs):
        return {"ok": True}

    def mark_starting(self):
        return {"ok": True}

    def mark_ready(self):
        return {"ok": True}

    def try_state(self):
        return {"starting": False, "runtime_alive": self.alive, "terminated": False}

    def events(self, limit=100):
        return []

    def logs(self, name="runtime", lines=200):
        self.log_requests.append((name, lines))
        return [
            "INFO starting vLLM API server",
            "RuntimeError: The NVIDIA driver on your system is too old "
            "(found version 12020). cuda>=13.0 required.",
        ][-lines:]


def test_a_dead_model_server_aborts_setup_with_its_log_and_cleans_up(engine, monkeypatch):
    import time as _time

    from gpu4agents.core.errors import RuntimeStartupError

    watchdog = _DeadRuntimeWatchdog()
    monkeypatch.setattr(engine, "beacon", lambda _state: watchdog)
    started = _time.time()
    with pytest.raises(RuntimeStartupError) as caught:
        engine.setup()
    assert _time.time() - started < 10, "must not wait out the 10m startup timeout"
    message = str(caught.value)
    assert "model server process exited" in message
    assert "NVIDIA driver on your system is too old" in message
    assert watchdog.log_requests == [("runtime", 30)]
    # cleanup_on_setup_failure: the dead instance is not left billing
    assert any(call[0] == "destroy" for call in engine.provider.calls)
    assert engine.store.load().instance_id is None


def test_an_unknown_runtime_liveness_does_not_abort(engine, monkeypatch):
    """None (no pid file yet) and True keep waiting on the health probe."""
    for alive in (None, True):
        watchdog = _DeadRuntimeWatchdog(alive=alive)
        monkeypatch.setattr(engine, "beacon", lambda _state, w=watchdog: w)
        state = engine.setup(wait=False)
        assert engine._runtime_exited(state) is None
        assert watchdog.log_requests == []


# -- gpu logs: the instance's own files through the watchdog -----------------


def test_logs_default_to_the_runtime_log_when_the_watchdog_answers(engine, monkeypatch):
    engine.setup(wait=False)
    watchdog = _DeadRuntimeWatchdog()
    monkeypatch.setattr(engine, "beacon", lambda _state: watchdog)
    source, lines = engine.read_logs(limit=1)
    assert source == "runtime"
    assert lines == [watchdog.logs()[-1]]
    assert engine.read_logs(limit=5, source="onstart")[0] == "onstart"
    assert watchdog.log_requests[-1] == ("onstart", 5)
    assert engine.read_logs(source="provider") == ("provider", ["line one", "line two"])


def test_logs_fall_back_to_the_provider_when_the_watchdog_is_unreachable(engine):
    from gpu4agents.core.errors import StateError

    engine.setup(wait=False)  # the fake's watchdog endpoint refuses connections
    assert engine.read_logs() == ("provider", ["line one", "line two"])
    with pytest.raises(StateError, match="cannot read the watchdog log"):
        engine.read_logs(source="watchdog")
    with pytest.raises(ConfigError, match="unknown log source"):
        engine.read_logs(source="kernel")


# -- a bad host costs one attempt, not the setup ------------------------------
# Live: one RTX 3090 host could not reach Docker Hub (TLS handshake timeout,
# Vast status FAILED), another sat in `loading` pulling a 10 GB image past the
# startup timeout. Both were cleaned up, and both failed the whole setup.


def _booting(write_config, monkeypatch, lifecycle=None, **overrides):
    """An engine whose setup waits for real, minus the watchdog and the sleeps."""
    config = write_config(
        lifecycle=dict({"watchdog": {"enabled": False}}, **(lifecycle or {})), **overrides
    )
    lines: list[str] = []
    engine = Engine(
        config,
        provider=FakeProvider(config, CredentialResolver(config.project_dir)),
        echo=lines.append,
    )
    health_waits: list[float] = []

    def _healthy(url, timeout, **_kwargs):
        health_waits.append(timeout)

    monkeypatch.setattr("gpu4agents.core.engine.wait_for_healthy", _healthy)
    monkeypatch.setattr("gpu4agents.core.engine.time.sleep", lambda _seconds: None)
    return engine, lines, health_waits


def _events(engine, name):
    return [r for r in engine.events.tail(limit=200) if r["event"] == name]


def test_a_host_that_fails_to_boot_is_replaced_by_another_machine(write_config, monkeypatch):
    engine, lines, _ = _booting(write_config, monkeypatch)
    engine.provider.fail_first_hosts(1)

    state = engine.setup()

    assert state.state == InstanceState.READY.value
    assert state.instance_id == "i-2" and state.instance.host_id == "m2"
    assert list(engine.provider.instances) == ["i-2"], "never two instances alive"
    assert ("destroy", "i-1") in engine.provider.calls
    assert engine.provider.searches[-1] == ["m1"]
    assert (
        "! host m1 failed (Error response from daemon: Get "
        '"https://registry-1.docker.io/v2/": net/http: TLS handshake timeout); '
        "trying another machine (2/3)"
    ) in lines
    [failed] = _events(engine, "host_failed")
    assert failed["instance_id"] == "i-1" and failed["host_id"] == "m1"
    assert failed["attempt"] == 1 and "TLS handshake timeout" in failed["reason"]
    destroyed = _events(engine, "instance_destroyed")
    assert [(r["instance_id"], r["reason"]) for r in destroyed] == [("i-1", "host_failure")]
    assert engine.store.load().instance_id == "i-2"


def test_setup_gives_up_after_host_attempts_and_leaves_nothing_running(
    write_config, monkeypatch
):
    from gpu4agents.core.errors import HostFailure

    engine, _, _ = _booting(write_config, monkeypatch)
    engine.provider.fail_first_hosts(3)

    with pytest.raises(HostFailure) as caught:
        engine.setup()

    message = str(caught.value)
    assert "3 machines in a row failed to boot" in message
    assert "compute.host_attempts" in message
    assert "TLS handshake timeout" in message
    assert engine.provider.instances == {}
    assert engine.store.load().instance_id is None
    assert engine.provider.rented_hosts == ["m1", "m2", "m3"]
    assert [r["attempt"] for r in _events(engine, "host_failed")] == [1, 2, 3]
    assert _events(engine, "setup_failed")
    from gpu4agents.core.hosts import BadHosts

    # The last host too, not only the two that were replaced.
    assert sorted(BadHosts().recent("fake")) == ["m1", "m2", "m3"]


def test_a_failed_machine_is_never_rented_again(write_config, monkeypatch):
    engine, _, _ = _booting(write_config, monkeypatch, compute={"host_attempts": 4})
    engine.provider.fail_first_hosts(3, how="vanish")

    state = engine.setup()

    assert engine.provider.rented_hosts == ["m1", "m2", "m3", "m4"]
    assert state.instance.host_id == "m4"
    assert engine.provider.searches[-1] == ["m1", "m2", "m3"]
    assert list(engine.provider.instances) == [state.instance_id]


def test_a_stuck_image_pull_times_out_on_boot_timeout_not_startup_timeout(
    write_config, monkeypatch
):
    engine, lines, health_waits = _booting(
        write_config, monkeypatch, health={"boot_timeout": "1s", "startup_timeout": "7m"}
    )
    engine.provider.fail_first_hosts(1, how="loading")

    state = engine.setup()

    assert state.instance.host_id == "m2"
    assert (
        "! host m1 failed (image pull timed out after 1s); trying another machine (2/3)"
        in lines
    )
    # The model server's own wait is startup_timeout, untouched by the boot.
    assert health_waits == [420.0]


def _with_clock(monkeypatch, start=1_000_000.0):
    """Make the boot wait's sleeps advance a fake clock: minutes in no time."""
    clock = [start]
    monkeypatch.setattr("gpu4agents.core.engine.time.time", lambda: clock[0])
    monkeypatch.setattr(
        "gpu4agents.core.engine.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    return clock


def test_a_frozen_image_pull_is_given_up_long_before_boot_timeout(write_config, monkeypatch):
    """Live: three hosts froze on one pull line until the 30-minute boot_timeout."""
    engine, lines, _ = _booting(write_config, monkeypatch)
    clock = _with_clock(monkeypatch)
    engine.provider.fail_first_hosts(1, how="loading")
    engine.provider.boot_message = lambda poll: (
        "995efd0b979d: Verifying Checksum" if poll < 20 else "067869a57bf4: Pull complete"
    )
    started = clock[0]

    state = engine.setup()

    assert state.instance.host_id == "m2"
    assert "! host m1 failed (image pull stalled for 10m); trying another machine (2/3)" in lines
    # Two minutes of progress, then ten without: replaced at about 12, not 30.
    assert 12 * 60 <= clock[0] - started < 13 * 60
    assert engine.provider.searches[-1] == ["m1"], "the stalled host is not rented again"


def test_a_pull_retrying_one_layer_counts_as_stalled(write_config, monkeypatch):
    engine, lines, _ = _booting(write_config, monkeypatch)
    clock = _with_clock(monkeypatch)
    engine.provider.fail_first_hosts(1, how="loading")
    engine.provider.boot_message = lambda poll: (
        f"60c68e5ca95a: Retrying in {1 + poll % 3} second{'s' if poll % 3 else ''}"
    )
    started = clock[0]

    engine.setup()

    assert "! host m1 failed (image pull stalled for 10m); trying another machine (2/3)" in lines
    assert 10 * 60 <= clock[0] - started < 11 * 60


def test_a_slow_pull_that_keeps_moving_waits_for_boot_timeout(write_config, monkeypatch):
    """A new layer every 8 minutes is slow, not stuck: only boot_timeout ends it."""
    engine, lines, _ = _booting(write_config, monkeypatch)
    clock = _with_clock(monkeypatch)
    engine.provider.fail_first_hosts(1, how="loading")
    engine.provider.boot_message = lambda poll: f"layer{poll * 6 // 480}: Pull complete"
    started = clock[0]

    engine.setup()

    assert "! host m1 failed (image pull timed out after 30m); trying another machine (2/3)" in lines
    assert clock[0] - started >= 30 * 60


def test_pull_stall_detection_can_be_turned_off(write_config, monkeypatch):
    engine, lines, _ = _booting(
        write_config, monkeypatch, health={"pull_stall_timeout": "off", "boot_timeout": "20m"}
    )
    _with_clock(monkeypatch)
    engine.provider.fail_first_hosts(1, how="loading")
    engine.provider.boot_message = lambda poll: "067869a57bf4: Pull complete"

    engine.setup()

    assert "! host m1 failed (image pull timed out after 20m); trying another machine (2/3)" in lines


def test_a_provider_that_reports_no_pull_progress_gets_the_whole_boot_timeout(write_config, monkeypatch):
    """RunPod says only STARTING while it pulls: ten quiet minutes are not a stall there."""
    from dataclasses import replace

    engine, lines, _ = _booting(write_config, monkeypatch)
    clock = _with_clock(monkeypatch)
    engine.provider.capabilities = replace(engine.provider.capabilities, boot_progress=False)
    engine.provider.fail_first_hosts(1, how="loading")
    engine.provider.boot_message = lambda poll: "STARTING"
    started = clock[0]

    engine.setup()

    assert "! host m1 failed (image pull timed out after 30m); trying another machine (2/3)" in lines
    assert clock[0] - started >= 30 * 60


def test_a_model_server_that_dies_is_not_blamed_on_the_host(engine, monkeypatch):
    from gpu4agents.core.errors import HostFailure, RuntimeStartupError

    monkeypatch.setattr(engine, "beacon", lambda _state: _DeadRuntimeWatchdog())
    with pytest.raises(RuntimeStartupError) as caught:
        engine.setup()

    assert not isinstance(caught.value, HostFailure)
    assert engine.provider.counter == 1, "the same config fails on every machine"
    assert not _events(engine, "host_failed")
    assert engine.provider.instances == {}


def test_each_replacement_offer_is_held_to_the_budget(write_config, monkeypatch):
    engine, _, _ = _booting(write_config, monkeypatch, budget={"max_hourly_cost_usd": 0.50})
    engine.provider.offer_price = 0.90
    engine.provider.host_prices = {"m1": 0.30}
    engine.provider.fail_first_hosts(1)

    with pytest.raises(BudgetExceededError):
        engine.setup()

    assert engine.provider.counter == 1
    assert engine.provider.instances == {}
    assert engine.store.load().instance_id is None


def test_without_cleanup_a_failed_host_is_kept_and_not_retried(write_config, monkeypatch):
    from gpu4agents.core.errors import HostFailure

    engine, _, _ = _booting(
        write_config, monkeypatch, lifecycle={"cleanup_on_setup_failure": False}
    )
    engine.provider.fail_first_hosts(1)

    with pytest.raises(HostFailure):
        engine.setup()

    assert engine.provider.counter == 1
    assert list(engine.provider.instances) == ["i-1"]


def test_the_remote_startup_grace_covers_the_boot_and_the_model_server(write_config):
    import base64
    import json
    import re
    import zlib

    def grace(**health):
        script = _engine(write_config, health=health)._build_onstart("t" * 43)
        blob = re.search(r"echo '(\S+)' > /etc/jambu/watchdog.json.b85", script).group(1)
        return json.loads(zlib.decompress(base64.b85decode(blob)))["startup_grace_s"]

    assert grace() == 30 * 60 + 10 * 60 + 300  # boot_timeout 30m by default
    assert grace(boot_timeout="20m", startup_timeout="5m") == 20 * 60 + 5 * 60 + 300


class _NetworkDeadRuntimeWatchdog(_DeadRuntimeWatchdog):
    """Live: vLLM died fetching weights - `httpx.RemoteProtocolError` from the hub."""

    def logs(self, name="runtime", lines=200):
        self.log_requests.append((name, lines))
        return [
            "  File \"huggingface_hub/utils/_http.py\", line 540, in _http_backoff_base",
            "httpx.RemoteProtocolError: Server disconnected without sending a response.",
        ][-lines:]


def test_a_model_download_killed_by_the_host_network_moves_to_another_host(engine, monkeypatch):
    from gpu4agents.core.errors import HostFailure

    monkeypatch.setattr(engine, "beacon", lambda _state: _NetworkDeadRuntimeWatchdog())
    with pytest.raises(HostFailure, match="Server disconnected"):
        engine.setup()

    assert engine.provider.counter == engine.config.compute.host_attempts
    assert len(_events(engine, "host_failed")) == engine.config.compute.host_attempts
    assert engine.provider.instances == {}


def test_failed_hosts_are_remembered_by_the_next_setup(write_config, monkeypatch):
    from gpu4agents.core.errors import HostFailure
    from gpu4agents.core.hosts import BadHosts

    engine, _, _ = _booting(write_config, monkeypatch)
    engine.provider.fail_first_hosts(1)
    engine.setup()
    assert BadHosts().recent("fake") == ["m1"]
    engine.destroy()

    # A fresh setup, even with no failures scripted, never rents m1 again.
    engine.provider.failing_hosts = set()
    engine.setup()
    hosts = [i.host_id for i in engine.provider.instances.values()]
    assert hosts and "m1" not in hosts


def test_bad_hosts_expire(tmp_path):
    from gpu4agents.core.hosts import BadHosts

    memory = BadHosts(tmp_path / "bad.json", ttl_s=60)
    memory.mark("vast", "150616", "image pull timed out", now=1000.0)
    assert memory.recent("vast", now=1030.0) == ["150616"]
    assert memory.recent("vast", now=1061.0) == []
    (tmp_path / "bad.json").write_text("{not json")
    assert memory.recent("vast") == []  # a corrupt file never blocks a setup


def test_status_reads_through_a_setup_holding_the_lock(engine, monkeypatch):
    """A setup holds the lock for the whole boot; status must not fail for 15 minutes."""
    from gpu4agents.core import engine as engine_mod
    from gpu4agents.core.errors import LockBusy
    from gpu4agents.core.state import StateStore

    engine.setup(wait=False)
    monkeypatch.setattr(engine_mod, "READ_LOCK_WAIT_S", 0.2)
    other = StateStore(engine.config.state_dir)
    with other.process_lock():
        payload = engine.status()
        assert payload["busy"] is True
        assert payload["instance_id"] == engine.store.load().instance_id
        with pytest.raises(LockBusy) as busy:
            with StateStore(engine.config.state_dir).process_lock(timeout=0.2):
                pass
        assert busy.value.exit_code == 12
    assert engine.status()["busy"] is False
