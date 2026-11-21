"""lifecycle.park: an idle machine keeps its disk, and setup restarts it."""

from __future__ import annotations

from fakes import FakeProvider

from gpu4agents.core import engine as engine_mod
from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.engine import Engine
from gpu4agents.core.errors import HostFailure
from gpu4agents.core.models import InstanceState


def _engine(write_config, monkeypatch, **lifecycle):
    config = write_config(lifecycle=lifecycle)
    engine = Engine(config, provider=FakeProvider(config, CredentialResolver(config.project_dir)))
    # Only the restart path is under test: no watchdog or model server to wait for.
    monkeypatch.setattr(engine, "_connect_watchdog", lambda state: None)
    monkeypatch.setattr(engine, "_wait_runtime", lambda state: None)
    return engine


def test_park_keeps_the_disk_for_its_duration(write_config):
    config = write_config(lifecycle={"park": "6h"})
    assert config.lifecycle.keep_disk is True
    assert config.stopped_ttl_s == 6 * 3600
    # Off by default: an idle machine where a stop bills is destroyed, as before.
    plain = write_config()
    assert plain.lifecycle.park is None and plain.lifecycle.keep_disk is False


def test_a_parked_machine_that_does_not_come_back_is_replaced(write_config, monkeypatch):
    engine = _engine(write_config, monkeypatch, park="6h")
    state = engine.setup(wait=False)
    first = state.instance_id
    engine.stop()
    assert engine.store.load().state == InstanceState.STOPPED.value

    def never_back(state, timeout=None):
        if timeout == engine_mod.PARK_RESTART_S:
            raise HostFailure("not running after 2m", reason="restart timed out", instance=state.instance)
        return original(state, timeout)

    original = engine._wait_running
    monkeypatch.setattr(engine, "_wait_running", never_back)
    state = engine.ensure_ready()
    assert state.instance_id and state.instance_id != first
    events = [e["event"] for e in engine.events.tail(50)]
    assert "park_restart_failed" in events
    assert first not in engine.provider.instances


def test_without_park_a_failed_restart_still_fails(write_config, monkeypatch):
    import pytest

    engine = _engine(write_config, monkeypatch)
    engine.setup(wait=False)
    engine.stop()

    def boom(state, timeout=None):
        raise HostFailure("not running", reason="restart timed out", instance=state.instance)

    monkeypatch.setattr(engine, "_wait_running", boom)
    with pytest.raises(HostFailure):
        engine.ensure_ready()
