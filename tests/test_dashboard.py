from __future__ import annotations

import json
import socket
import threading
import time

import httpx
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


def test_select_instances_includes_error_rows():
    # The page renders GPUs from `instances`; an error row must surface there,
    # not vanish because it has no instance_id.
    from gpu4agents.dashboard.collect import select_instances

    rows = [
        {"project": "a", "instance_id": "i-1"},
        {"project": "b", "error": "corrupt"},
        {"project": "c"},  # destroyed/no instance: not a GPU to show
    ]
    picked = select_instances(rows)
    assert {r["project"] for r in picked} == {"a", "b"}


def test_live_project_paths_drops_missing(tmp_path):
    # Dead registry entries (ephemeral job dirs, deleted projects) must not
    # appear as controllable/rentable targets.
    from gpu4agents.dashboard.collect import live_project_paths

    a = tmp_path / "a" / "gpu.yaml"
    a.parent.mkdir(parents=True)
    a.write_text("gpu_runtime: {}\n")
    missing = tmp_path / "gone" / "gpu.yaml"
    out = live_project_paths([a, missing])
    assert out == [a]


def test_reconcile_marks_gone_when_not_live():
    # The truth is the provider's live instance list; a local state.json that
    # still names a dead instance is a ghost and must be marked gone.
    from gpu4agents.dashboard.collect import reconcile

    projects = [
        {"project": "a", "instance_id": "i-live"},
        {"project": "b", "instance_id": "i-dead"},
        {"project": "c"},  # no instance to reconcile
        {"project": "d", "error": "corrupt"},
    ]
    out = reconcile(projects, live_ids={"i-live"})
    by = {r["project"]: r for r in out}
    assert by["a"]["gone"] is False and by["a"]["verified"] is True
    assert by["b"]["gone"] is True and by["b"]["state"] == "gone"
    assert by["c"]["verified"] is True
    assert "gone" not in by["d"]  # an error row is never reconciled


def test_reconcile_unverified_when_no_live_data():
    from gpu4agents.dashboard.collect import reconcile

    out = reconcile([{"project": "a", "instance_id": "i"}], live_ids=None)
    assert out[0]["verified"] is False
    assert "gone" not in out[0]


def test_select_instances_drops_ghosts():
    from gpu4agents.dashboard.collect import select_instances

    rows = [{"project": "a", "instance_id": "i"},
            {"project": "b", "instance_id": "j", "gone": True}]
    assert [r["project"] for r in select_instances(rows)] == ["a"]


# --- ops: shell to the gpu CLI (live truth, logs, offers, control) ---------

def test_ops_live_ids_parses_and_tolerates_failure():
    from gpu4agents.dashboard import ops

    def ok(args, cwd=None, timeout=0):
        # `gpu --json ls` keys the id as "instance_id" (not "id").
        return (0, json.dumps({"provider": "vast",
                               "instances": [{"instance_id": "60697697", "state": "ready"},
                                             {"instance_id": 77}]}), "")

    assert ops.live_ids(run=ok) == {"60697697", "77"}

    def fail(args, cwd=None, timeout=0):
        return (2, "", "no credentials")

    assert ops.live_ids(run=fail) is None  # unverifiable → None, never a crash


def test_ops_offers_parses():
    from gpu4agents.dashboard import ops

    def run(args, cwd=None, timeout=0):
        return (0, json.dumps({"offers": [{"id": "1", "gpu": "A100", "hourly_cost_usd": 0.9}]}), "")

    offers = ops.offers(run=run)
    assert offers[0]["id"] == "1"


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


# --- Task 4: server + token gate -------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _boot(cfg, provider, gate):
    from gpu4agents.dashboard.probe import ReachabilityWatch
    from gpu4agents.dashboard.server import DashboardServer

    watch = ReachabilityWatch(interval_s=15, history_s=300)
    server = DashboardServer(cfg, provider, watch, gate, log=lambda _m: None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.2)
    return server


def test_token_gate_accepts_bearer_and_cookie_rejects_absent():
    from gpu4agents.dashboard.auth import TokenGate

    gate = TokenGate("SEKRET")
    assert gate.check({}, "") is False
    assert gate.check({"Authorization": "Bearer SEKRET"}, "") is True
    assert gate.check({}, "token=SEKRET") is True
    assert gate.check({"Cookie": "gpu_dash=SEKRET"}, "") is True
    assert gate.check({"Authorization": "Bearer nope"}, "") is False


def test_api_state_shape():
    from gpu4agents.core.config import DashboardConfig
    from gpu4agents.dashboard.auth import TokenGate

    cfg = DashboardConfig(listen=f"127.0.0.1:{_free_port()}")
    provider = lambda: {  # noqa: E731
        "instances": [],
        "projects": [{"project": "p", "endpoint": "http://x", "model": "m"}],
        "proxies": [],
    }
    server = _boot(cfg, provider, TokenGate(None))
    try:
        r = httpx.get(f"http://{cfg.listen}/api/state", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert {"instances", "projects", "proxies", "probes", "generated_at", "control"} <= set(body)
        assert body["control"] is False  # read-only by default
        assert "api_key" not in json.dumps(body)
    finally:
        server.stop()


def test_api_state_requires_token_when_set():
    from gpu4agents.core.config import DashboardConfig
    from gpu4agents.dashboard.auth import TokenGate

    cfg = DashboardConfig(listen=f"127.0.0.1:{_free_port()}")
    provider = lambda: {"instances": [], "projects": [], "proxies": []}  # noqa: E731
    server = _boot(cfg, provider, TokenGate("T"))
    base = f"http://{cfg.listen}"
    try:
        assert httpx.get(f"{base}/api/state", timeout=5).status_code == 401
        assert httpx.get(f"{base}/api/state", headers={"Authorization": "Bearer T"},
                         timeout=5).status_code == 200
        r = httpx.get(f"{base}/api/state?token=T", timeout=5)
        assert r.status_code == 200
        assert "gpu_dash=T" in r.headers.get("set-cookie", "")
    finally:
        server.stop()


# --- Task 5: CLI -----------------------------------------------------------

def test_cli_dashboard_start_help():
    from typer.testing import CliRunner

    from gpu4agents.cli.dashboard import dashboard_app

    res = CliRunner().invoke(dashboard_app, ["--help"])
    assert res.exit_code == 0
    assert "start" in res.output


# --- Task 6: full page + non-loopback bind gate ----------------------------

def test_non_loopback_without_token_refuses():
    from gpu4agents.dashboard.server import guard_bind

    with pytest.raises(ValueError):
        guard_bind("0.0.0.0", None, required=True)
    guard_bind("0.0.0.0", "T", required=True)     # a token makes it ok
    guard_bind("127.0.0.1", None, required=True)  # loopback is always ok


def test_page_served_and_selfcontained():
    import re

    from gpu4agents.core.config import DashboardConfig
    from gpu4agents.dashboard.auth import TokenGate

    cfg = DashboardConfig(listen=f"127.0.0.1:{_free_port()}")
    server = _boot(cfg, lambda: {"instances": [], "projects": [], "proxies": []}, TokenGate(None))
    try:
        html = httpx.get(f"http://{cfg.listen}/", timeout=5).text
        assert "<" in html and "gpu dashboard" in html.lower()
        assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)
    finally:
        server.stop()


# --- Task 7: systemd service -----------------------------------------------

def test_service_unit_is_restarting_and_boot_safe(tmp_path):
    from gpu4agents.dashboard import service as svc

    unit = svc.render_unit(project_dir=tmp_path, python="/usr/bin/python3")
    assert "Type=simple" in unit
    assert "Restart=always" in unit
    assert "WantedBy=default.target" in unit
    assert "dashboard" in unit and "gpu4agents" in unit


def test_service_slug_per_project(tmp_path):
    from gpu4agents.dashboard import service as svc

    a = svc.unit_name(tmp_path / "proj-a")
    b = svc.unit_name(tmp_path / "proj-b")
    assert a != b
    assert a == svc.unit_name(tmp_path / "proj-a")


# --- Task 8: control actions (phase 3) -------------------------------------

def _boot_control(cfg, actions, gate, project_dirs=None):
    from gpu4agents.dashboard.probe import ReachabilityWatch
    from gpu4agents.dashboard.server import DashboardServer

    watch = ReachabilityWatch(interval_s=15, history_s=300)
    provider = lambda: {"instances": [], "projects": [], "proxies": []}  # noqa: E731
    server = DashboardServer(cfg, provider, watch, gate, log=lambda _m: None,
                             actions=actions, project_dirs=project_dirs if project_dirs is not None else {"p": "/x"})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.2)
    return server


def test_control_requires_token_and_confirm():
    from gpu4agents.core.config import DashboardConfig
    from gpu4agents.dashboard.auth import TokenGate

    calls = []
    actions = {"stop": lambda body: calls.append(("stop", body.get("project"))) or {"stopped": body.get("project")}}
    cfg = DashboardConfig(listen=f"127.0.0.1:{_free_port()}", control_enabled=True)
    server = _boot_control(cfg, actions, TokenGate("T"), project_dirs={"p": "/x"})
    base = f"http://{cfg.listen}"
    hdr = {"Authorization": "Bearer T", "Content-Type": "application/json"}
    try:
        # no token → 401, nothing called
        assert httpx.post(f"{base}/api/control/stop",
                          content=json.dumps({"project": "p", "confirm": True}),
                          headers={"Content-Type": "application/json"}, timeout=5).status_code == 401
        # token but no confirm → 400
        assert httpx.post(f"{base}/api/control/stop",
                          content=json.dumps({"project": "p"}), headers=hdr, timeout=5).status_code == 400
        assert calls == []
        # token + confirm → 200, action called once
        r = httpx.post(f"{base}/api/control/stop",
                       content=json.dumps({"project": "p", "confirm": True}), headers=hdr, timeout=5)
        assert r.status_code == 200
        assert calls == [("stop", "p")]
    finally:
        server.stop()


def test_logs_and_offers_routes():
    from gpu4agents.core.config import DashboardConfig
    from gpu4agents.dashboard.auth import TokenGate
    from gpu4agents.dashboard.probe import ReachabilityWatch
    from gpu4agents.dashboard.server import DashboardServer

    cfg = DashboardConfig(listen=f"127.0.0.1:{_free_port()}")
    provider = lambda: {"instances": [], "projects": [], "proxies": []}  # noqa: E731
    watch = ReachabilityWatch(interval_s=15, history_s=300)
    server = DashboardServer(cfg, provider, watch, TokenGate(None), log=lambda _m: None,
                             logs_fn=lambda p: f"logs for {p}",
                             offers_fn=lambda project: [{"id": "1", "for": project}])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.2)
    base = f"http://{cfg.listen}"
    try:
        r = httpx.get(f"{base}/api/logs?project=x", timeout=5)
        assert r.status_code == 200 and r.text == "logs for x"
        o = httpx.get(f"{base}/api/offers?project=deepseek-qwen", timeout=5)
        assert o.status_code == 200 and o.json()["offers"][0]["for"] == "deepseek-qwen"
    finally:
        server.stop()


def test_logs_route_404_without_fn():
    from gpu4agents.core.config import DashboardConfig
    from gpu4agents.dashboard.auth import TokenGate

    cfg = DashboardConfig(listen=f"127.0.0.1:{_free_port()}")
    server = _boot(cfg, lambda: {"instances": [], "projects": [], "proxies": []}, TokenGate(None))
    try:
        assert httpx.get(f"http://{cfg.listen}/api/logs?project=x", timeout=5).status_code == 404
        assert httpx.get(f"http://{cfg.listen}/api/offers", timeout=5).status_code == 404
    finally:
        server.stop()


def test_control_rejects_foreign_project():
    # A control action names its project; one missing from project_dirs (not a
    # project this host manages) gets 409 and nothing runs, a known one runs.
    from gpu4agents.core.config import DashboardConfig
    from gpu4agents.dashboard.auth import TokenGate

    calls = []
    actions = {"destroy": lambda body: calls.append(body.get("project"))}
    cfg = DashboardConfig(listen=f"127.0.0.1:{_free_port()}", control_enabled=True)
    server = _boot_control(cfg, actions, TokenGate("T"), project_dirs={"local": "/x"})
    base = f"http://{cfg.listen}"
    hdr = {"Authorization": "Bearer T", "Content-Type": "application/json"}
    try:
        r = httpx.post(f"{base}/api/control/destroy",
                       content=json.dumps({"project": "other", "confirm": True}), headers=hdr, timeout=5)
        assert r.status_code == 409
        assert calls == []  # an unknown project is refused, nothing ran
        r2 = httpx.post(f"{base}/api/control/destroy",
                        content=json.dumps({"project": "local", "confirm": True}), headers=hdr, timeout=5)
        assert r2.status_code == 200
        assert calls == ["local"]
    finally:
        server.stop()


def test_control_disabled_returns_404():
    from gpu4agents.core.config import DashboardConfig
    from gpu4agents.dashboard.auth import TokenGate

    cfg = DashboardConfig(listen=f"127.0.0.1:{_free_port()}", control_enabled=False)
    server = _boot_control(cfg, {"stop": lambda project: None}, TokenGate("T"))
    base = f"http://{cfg.listen}"
    try:
        r = httpx.post(f"{base}/api/control/stop",
                       content=json.dumps({"project": "p", "confirm": True}),
                       headers={"Authorization": "Bearer T", "Content-Type": "application/json"}, timeout=5)
        assert r.status_code == 404
    finally:
        server.stop()
