from __future__ import annotations

import json

import pytest

from gpu4agents.core.config import DashboardConfig, RuntimeConfig
from gpu4agents.core.state import RuntimeState, StateStore

from conftest import BASE_CONFIG


def _write_project(tmp_path, name, **state_kw):
    cfg = tmp_path / name / "gpu.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("gpu_runtime: {}\n")
    store = StateStore(cfg.parent / ".gpu")
    store.ensure_dirs()
    store.save(RuntimeState(**state_kw))
    return cfg


# --- Task 1: config --------------------------------------------------------

def test_config_defaults_dashboard_off():
    rc = RuntimeConfig.model_validate(BASE_CONFIG)
    assert rc.dashboard.enabled is False
    assert rc.dashboard.listen_host_port == ("127.0.0.1", 8137)


def test_dashboard_bad_listen_rejected():
    with pytest.raises(ValueError):
        DashboardConfig(listen="nope").listen_host_port


# --- Task 2: collectors ----------------------------------------------------

def test_project_snapshot_reads_state(tmp_path):
    from gpu4agents.dashboard.collect import project_snapshots

    cfg = _write_project(
        tmp_path, "proj-a", instance_id="i-1", state="running",
        model_id="org/Model-27B", endpoint="http://1.2.3.4:9/",
        hourly_cost_usd=0.9, session_cost_usd=2.1,
        last_activity=1000.0, api_key="SECRET",
    )
    [snap] = project_snapshots([cfg], now=1300.0)
    assert snap["project"] == "proj-a"
    assert snap["instance_id"] == "i-1"
    assert snap["model"] == "org/Model-27B"
    assert snap["endpoint"] == "http://1.2.3.4:9/"
    assert snap["hourly_cost_usd"] == 0.9
    assert snap["idle_seconds"] == 300.0
    assert "api_key" not in snap
    assert "SECRET" not in json.dumps(snap)


def test_corrupt_state_becomes_error_row(tmp_path):
    from gpu4agents.dashboard.collect import project_snapshots

    cfg = tmp_path / "proj-b" / "gpu.yaml"
    (cfg.parent / ".gpu").mkdir(parents=True)
    (cfg.parent / ".gpu" / "state.json").write_text("{not json")
    [snap] = project_snapshots([cfg], now=0.0)
    assert snap["project"] == "proj-b"
    assert "error" in snap


# --- Task 3: reachability watch --------------------------------------------

def test_probe_records_failure_window():
    from gpu4agents.dashboard.probe import ReachabilityWatch

    w = ReachabilityWatch(interval_s=15, history_s=300)
    w.record("http://e", (True, 10.0, None), ts=1000.0)
    w.record("http://e", (False, None, "[Errno 110] Connection timed out"), ts=1005.0)
    w.record("http://e", (True, 12.0, None), ts=1010.0)
    snap = w.snapshot("http://e")
    assert snap["failures_in_window"] == 1
    assert snap["errno110_in_window"] == 1
    assert snap["last_ok"] is True
    assert snap["last_connect_ms"] == 12.0


def test_window_trims_old():
    from gpu4agents.dashboard.probe import ReachabilityWatch

    w = ReachabilityWatch(interval_s=15, history_s=100)
    w.record("http://e", (True, 10.0, None), ts=1000.0)
    w.record("http://e", (True, 11.0, None), ts=1200.0)  # 200s later → first drops
    snap = w.snapshot("http://e")
    assert len(snap["window"]) == 1
