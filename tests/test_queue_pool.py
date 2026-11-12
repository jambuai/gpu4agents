"""The pool: machines rented from one gpu.yaml, opened without blocking,
health-checked before work, watched, and retired.

Each rented machine is a FakeProvider instance whose watchdog is a real
watchdog process (the ``runner`` fixture), so health checks, dispatch and
deadlines travel over HTTP exactly as on a GPU machine.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest
import test_watchdog

from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.errors import CapacityUnavailableError
from gpu4agents.core.events import EventLog
from gpu4agents.core.hosts import BadHosts
from gpu4agents.core.job import build_bundle
from gpu4agents.core.models import Endpoint
from gpu4agents.queue.observe import Observer
from gpu4agents.queue.pool import HealthResult, Pool, RetireResult, builtin_health_command
from gpu4agents.queue.pool_types import MachineState
from gpu4agents import watchdog as watchdog_pkg

from fakes import FakeProvider

TOKEN = "pool-test-token-0123456789abcdef0123456789"


class Clock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class PoolProvider(FakeProvider):
    """Each rented machine gets its own live watchdog."""

    def __init__(self, config, credentials, runner) -> None:
        super().__init__(config, credentials)
        self.runner = runner
        self.agents: dict = {}
        self.setups = 0
        self.lock = threading.Lock()
        self.before_setup = None  # called (outside the lock) before each rent

    def setup(self, spec):
        if self.before_setup is not None:
            self.before_setup()
        with self.lock:  # the fake and the runner fixture are not thread-safe
            return self._setup(spec)

    def _setup(self, spec):
        self.setups += 1
        instance = super().setup(spec)
        agent, _ = self.runner()
        self.agents[instance.id] = agent
        port = agent.port
        instance.endpoints["watchdog"] = Endpoint("watchdog", f"http://127.0.0.1:{port}", 8777, port)
        return instance


@pytest.fixture
def pool_parts(write_config, runner, tmp_path, monkeypatch):
    monkeypatch.setattr(test_watchdog, "generate_token", lambda: TOKEN)
    monkeypatch.setattr(watchdog_pkg, "generate_token", lambda: TOKEN)

    def _make(health_command="true", **queue):
        q = {"health_check": {"command": health_command, "timeout": "30s"}, "max_machines": 3}
        q.update(queue)
        cfg = write_config(runtime={"engine": "none"}, queue=q)
        provider = PoolProvider(cfg, CredentialResolver(cfg.project_dir), runner)
        observer = Observer(EventLog(tmp_path / "logs"), tmp_path / "metrics.prom")
        clock = Clock()
        pool = Pool(cfg, observer, provider=provider, clock=clock)
        return pool, provider, observer, clock, cfg

    return _make


def events(tmp_path: Path, name: str) -> list[dict]:
    path = tmp_path / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["event"] == name]


def settle(pool: Pool, m, timeout: float = 30.0):
    """Poll until the machine has left provisioning/checking."""
    end = time.time() + timeout
    while time.time() < end:
        pool.poll()
        thread = pool._threads.get(m.slot)
        if m.state not in (MachineState.PROVISIONING, MachineState.CHECKING) and not (thread and thread.is_alive()):
            return m
        time.sleep(0.1)
    raise AssertionError(f"{m.slot} still {m.state}")


def wait_remote(pool: Pool, m, rid: str, timeout: float = 20.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        info = pool.job_info(m, rid)
        if info["state"] != "running":
            return info
        time.sleep(0.1)
    raise AssertionError(f"{rid} still running")


def make_bundle(tmp_path: Path, files: dict[str, str]) -> Path:
    src = tmp_path / "jobsrc"
    for name, text in files.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_text(text)
    bundle = tmp_path / "bundle.tar.gz"
    build_bundle(src, [], bundle)
    return bundle


# -- the tests the brief names ------------------------------------------------


def test_open_returns_at_once_and_becomes_ready_after_the_health_check(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    started = time.time()
    m = pool.open()
    assert time.time() - started < 1.0
    assert m.state == MachineState.PROVISIONING and m.slot == "s1"
    assert [x.slot for x in pool.machines()] == ["s1"]

    settle(pool, m)
    assert m.state == MachineState.READY
    assert m.instance_id and m.ready_at is not None and m.hourly_usd == pytest.approx(0.42)
    agent = provider.agents[m.instance_id]
    health_jobs = [j for j in pool._jobs(m).list() if j["id"].startswith("pool-health-")]
    assert len(health_jobs) == 1 and health_jobs[0]["state"] == "succeeded"
    # Started with a deadline that covers boot and startup, set before any work.
    assert agent.client.state()["deadline_at"] == pytest.approx(m.deadline_at)
    assert m.deadline_at >= clock() + cfg.boot_timeout_s
    assert events(tmp_path, "pool_machine_requested") and events(tmp_path, "pool_machine_ready")


def test_a_failed_health_check_retires_and_marks_the_host_bad(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts(health_command="echo no gpu; exit 3")
    m = pool.open()
    settle(pool, m)
    assert m.state == MachineState.RETIRED
    assert pool.machines() == []
    assert provider.instances == {}  # destroyed
    assert "m1" in BadHosts().recent("fake")
    unhealthy = events(tmp_path, "pool_machine_unhealthy")
    assert unhealthy and "no gpu" in unhealthy[0]["detail"]
    assert events(tmp_path, "pool_machine_retired")
    assert observer.snapshot()["gpu_pool_health_failures_total"] == 1


def test_unresponsive_after_the_grace_period(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    provider.agents[m.instance_id].close()  # the watchdog stops answering
    clock.advance(10)
    pool.poll()
    assert m.state == MachineState.READY  # still within unresponsive_after
    clock.advance(301)
    pool.poll()
    assert m.state == MachineState.UNRESPONSIVE
    assert events(tmp_path, "pool_machine_unresponsive")
    assert m.instance_id in provider.instances  # the coordinator retires it, not poll


def test_a_vanished_machine_is_gone(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    provider.instances.pop(m.instance_id)
    pool.poll()
    assert m.state == MachineState.GONE
    assert events(tmp_path, "pool_machine_gone")
    result = pool.retire(m, "gone")
    assert m.state == MachineState.RETIRED and pool.machines() == []
    assert result.destroyed and BadHosts().recent("fake") == []  # vanishing is not the host's fault


def test_dispatch_twice_returns_already_running(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    bundle = make_bundle(tmp_path, {"main.py": "import os, pathlib, time\n"
                                    "p = pathlib.Path(os.environ['JAMBU_OUTPUT_DIR'])\n"
                                    "(p / 'out.txt').write_text(os.environ['GREETING'])\n"
                                    "time.sleep(1)\n"})
    manifest = {"run": "python3 main.py", "env": {"GREETING": "from-manifest"}}
    assert pool.dispatch(m, "q1-a1", bundle, manifest, {"GREETING": "hi"}) == "started"
    assert pool.dispatch(m, "q1-a1", bundle, manifest, {"GREETING": "hi"}) == "already_running"
    pool.poll()
    assert m.state == MachineState.BUSY and m.jobs == ["q1-a1"]
    assert [j["id"] for j in pool.remote_jobs(m)] == ["q1-a1"]  # health jobs are not queue jobs

    info = wait_remote(pool, m, "q1-a1")
    assert info["state"] == "succeeded"
    assert pool.fetch_outputs(m, "q1-a1", tmp_path / "back") == ["out.txt"]
    assert (tmp_path / "back" / "out.txt").read_text() == "hi"
    pool.poll()
    assert m.state == MachineState.READY and m.jobs == []


def test_extend_deadline_reaches_the_watchdog(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    until = m.deadline_at + 3600
    pool.extend_deadline(m, until)
    assert m.deadline_at == pytest.approx(until)
    assert provider.agents[m.instance_id].client.state()["deadline_at"] == pytest.approx(until)
    assert events(tmp_path, "pool_deadline_extended")
    # Never moved back.
    pool.extend_deadline(m, until - 1000)
    assert m.deadline_at == pytest.approx(until)


def test_no_capacity_backs_off_and_keeps_jobs_queued(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    provider.no_offers = True
    first = pool.open()
    settle(pool, first)
    assert pool.machines() == []  # nothing rented, nothing failed: the jobs just wait
    assert provider.setups == 1
    clock.advance(10)
    assert pool.open() is None
    clock.advance(10)
    assert pool.open() is None
    assert provider.setups == 1
    assert pool.open_backoff_s == 30 and pool.last_open_at is not None
    blocked = events(tmp_path, "pool_cap_blocked")
    assert len(blocked) == 1 and blocked[0]["cap"] == "provider_capacity"

    clock.advance(11)  # 31 s after the first attempt
    settle(pool, pool.open())
    assert provider.setups == 2
    assert pool.open_backoff_s == 60  # doubled
    assert len(events(tmp_path, "pool_cap_blocked")) == 1  # once per change

    provider.no_offers = False
    clock.advance(61)
    m = settle(pool, pool.open())
    assert m.state == MachineState.READY
    assert pool.open_backoff_s == 0


def test_backoff_is_capped_at_five_minutes(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    provider.no_offers = True
    for _ in range(8):
        m = pool.open()
        assert m is not None
        settle(pool, m)
        clock.advance(pool.open_backoff_s + 1)
    assert pool.open_backoff_s == 300


def test_slots_have_their_own_state_and_labels(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    a = pool.open()
    clock.advance(1)
    b = pool.open()
    settle(pool, a)
    settle(pool, b)
    assert [m.slot for m in pool.machines()] == ["s1", "s2"]  # oldest first
    state_dir = cfg.state_dir
    for m in (a, b):
        assert (state_dir / "pool" / m.slot / "state.json").exists()
        saved = json.loads((state_dir / "pool" / m.slot / "pool.json").read_text())
        assert saved["slot"] == m.slot and saved["state"] == MachineState.READY
        label = provider.instances[m.instance_id].label
        assert re.fullmatch(rf"jambu-.+-{m.slot}-{cfg.fingerprint()[:6]}", label)
    assert a.instance_id != b.instance_id

    pool.retire(a, "test")
    clock.advance(1)
    c = settle(pool, pool.open())
    assert c.slot == "s1"  # reused


def test_cold_start_is_measured(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    m = pool.open()
    clock.advance(42)
    settle(pool, m)
    snap = observer.snapshot()
    assert snap["gpu_pool_cold_start_seconds_count"] == 1
    assert snap["gpu_pool_cold_start_seconds_sum"] >= 42
    assert snap['gpu_pool_machines{state="ready"}'] == 1


# -- what the coordinator and the auditor lean on --------------------------------


def drop_remote_deadline(agent) -> None:
    """Restart a machine's watchdog without the deadline it held."""
    agent.close()
    state = json.loads(agent.state_path.read_text())
    state.pop("deadline_at", None)
    agent.state_path.write_text(json.dumps(state))
    agent.start()
    assert agent.client.state()["deadline_at"] is None


def test_a_restarted_pool_reloads_its_machines_and_resets_their_deadline(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    agent = provider.agents[m.instance_id]
    drop_remote_deadline(agent)
    again = Pool(cfg, observer, provider=provider, clock=clock)
    [n] = again.machines()
    assert (n.slot, n.instance_id, n.state) == ("s1", m.instance_id, MachineState.READY)
    again.poll()
    assert agent.client.state()["deadline_at"] >= clock() + cfg.boot_timeout_s


def test_dispatch_sets_a_deadline_on_an_adopted_machine_first(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    agent = provider.agents[m.instance_id]
    drop_remote_deadline(agent)
    again = Pool(cfg, observer, provider=provider, clock=clock)
    [n] = again.machines()
    bundle = make_bundle(tmp_path, {"main.py": "print('x')\n"})
    assert again.dispatch(n, "q2-a1", bundle, {"run": "python3 main.py"}, {}) == "started"
    assert agent.client.state()["deadline_at"] is not None


def test_orphans_are_this_project_s_untracked_slot_instances(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    fp = cfg.fingerprint()[:6]
    tracked = provider.instances[m.instance_id].label
    prefix = tracked[: tracked.index("-s1-") + 1]
    from gpu4agents.core.models import Instance

    def add(iid, label):
        provider.instances[iid] = Instance(id=iid, provider="fake", label=label, created_at=time.time() - 1000)

    add("x1", f"{prefix}s7-{fp}")          # orphan
    add("x2", f"{prefix}s8-ffffff")        # other config
    add("x3", f"jambu-other-s1-{fp}")       # other project
    add("x4", f"{prefix[:-1]}-{fp}")        # agent-mode machine, no slot
    assert [i.id for i in pool.orphans()] == ["x1"]


def test_keepalive_touches_the_watchdog(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    agent = provider.agents[m.instance_id]
    before = agent.client.state()["last_activity"]
    time.sleep(0.05)
    pool.keepalive(m)
    assert agent.client.state()["last_activity"] > before


def test_cancel_reaches_the_remote_job(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    bundle = make_bundle(tmp_path, {"main.py": "import time; time.sleep(60)\n"})
    pool.dispatch(m, "q3-a1", bundle, {"run": "python3 main.py"}, {})
    pool.cancel(m, "q3-a1")
    assert wait_remote(pool, m, "q3-a1")["state"] == "cancelled"


def test_poll_records_spend_per_machine(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    clock.advance(3600)
    pool.poll()
    assert pool.spend.total() == pytest.approx(0.42, rel=0.01)
    snap = observer.snapshot()
    assert snap["gpu_pool_usd_per_hour"] == pytest.approx(0.42)
    assert snap["gpu_pool_spend_usd_total"] == pytest.approx(pool.spend.total())


def test_the_built_in_health_check_tests_cuda_and_free_memory():
    cmd = builtin_health_command(None)
    assert "torch.cuda.is_available()" in cmd and "nvidia-smi -L" in cmd
    assert "memory.free" not in cmd
    cmd = builtin_health_command(None, min_free_vram_gb=10)
    assert "memory.free" in cmd and "10240" in cmd
    assert builtin_health_command("pytest -q smoke") .startswith("pytest -q smoke")


def test_health_result_shape():
    r = HealthResult(ok=False, detail="x")
    assert (r.ok, r.detail) == (False, "x")


# -- fix round 1 ----------------------------------------------------------------


def test_each_interval_is_billed_once_across_a_restart(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    provider.agents[m.instance_id].close()
    clock.advance(400)
    pool.poll()  # unresponsive now: skipped by reconcile, still billed
    assert m.state == MachineState.UNRESPONSIVE
    clock.advance(3200)
    pool.poll()
    billed = pool.spend.total()
    again = Pool(cfg, observer, provider=provider, clock=clock)
    again.poll()
    assert again.spend.total() == pytest.approx(billed)
    clock.advance(3600)
    again.poll()
    assert again.spend.total() == pytest.approx(billed + 0.42)


def test_a_bring_up_that_crashes_after_the_rent_leaves_the_machine_unhealthy(pool_parts, tmp_path, monkeypatch):
    pool, provider, observer, clock, cfg = pool_parts()

    def broken(m):
        raise OSError("disk full")

    monkeypatch.setattr(pool, "health_check", broken)
    m = settle(pool, pool.open())
    assert m.state == MachineState.UNHEALTHY and m.instance_id in provider.instances
    assert any("OSError" in e["detail"] for e in events(tmp_path, "pool_machine_unhealthy"))
    err = [e for e in events(tmp_path, "provider_error") if e.get("operation") == "pool_bring_up"]
    assert err and err[0]["error_type"] == "OSError"
    assert pool.retire(m, "bring-up crashed").destroyed
    assert provider.instances == {}


def test_resumes_are_capped(pool_parts, tmp_path, monkeypatch):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    with pool._lock:
        m.state = MachineState.CHECKING  # as if every bring-up died before finishing
    monkeypatch.setattr(pool, "_start", lambda m, fresh: None)
    for _ in range(4):
        pool.poll()
    assert m.state == MachineState.UNHEALTHY
    assert any("attempts" in e["detail"] for e in events(tmp_path, "pool_machine_unhealthy"))


def write_result(tmp_path):
    return make_bundle(tmp_path, {"main.py": "import os, pathlib\n"
                                  "(pathlib.Path(os.environ['JAMBU_OUTPUT_DIR']) / 'r.txt').write_text('42')\n"})


def test_retire_reports_what_it_saved(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    pool.dispatch(m, "q5-a1", write_result(tmp_path), {"run": "python3 main.py"}, {})
    wait_remote(pool, m, "q5-a1")
    pool.poll()
    assert m.jobs == ["q5-a1"]  # finished, outputs only on the machine
    assert m.state == MachineState.READY
    result = pool.retire(m, "test")
    assert isinstance(result, RetireResult) and result.destroyed and result.lost == []
    assert (result.pulled["q5-a1"] / "r.txt").read_text() == "42"
    [retired] = events(tmp_path, "pool_machine_retired")
    assert retired["outputs_pulled"] == 1 and retired["outputs_pulled_ids"] == ["q5-a1"]
    assert retired["outputs_lost"] == 0


def test_retire_destroys_even_when_outputs_cannot_be_saved(pool_parts, tmp_path, monkeypatch):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    pool.dispatch(m, "q6-a1", write_result(tmp_path), {"run": "python3 main.py"}, {})
    wait_remote(pool, m, "q6-a1")

    def no_space(self, job_id, dest):
        raise OSError("no space left")

    monkeypatch.setattr("gpu4agents.core.job.JobClient.download_outputs", no_space)
    result = pool.retire(m, "test")
    assert result.destroyed and result.lost == ["q6-a1"] and result.pulled == {}
    assert provider.instances == {}


def test_retire_counts_an_unreachable_machine_s_jobs_as_lost(pool_parts, tmp_path):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    pool.dispatch(m, "q7-a1", write_result(tmp_path), {"run": "sleep 30"}, {})
    provider.agents[m.instance_id].close()
    result = pool.retire(m, "unresponsive")
    assert result.destroyed and result.lost == ["q7-a1"]
    assert events(tmp_path, "pool_machine_retired")[0]["outputs_lost_ids"] == ["q7-a1"]


def test_backoff_runs_from_the_failure_not_the_open(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    provider.no_offers = True
    provider.before_setup = lambda: clock.advance(40)  # a slow rent that then fails
    settle(pool, pool.open())
    assert provider.setups == 1
    clock.advance(5)  # 45 s after the open, 5 s after the failure
    assert pool.open() is None and pool.backing_off()
    clock.advance(26)
    assert pool.open() is not None


def test_concurrent_failures_double_the_backoff_once(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    provider.no_offers = True
    gate = threading.Event()
    provider.before_setup = gate.wait
    a = pool.open()
    b = pool.open()
    gate.set()
    settle(pool, a)
    settle(pool, b)
    assert provider.setups == 2 and pool.open_backoff_s == 30


def test_a_reloaded_pool_waits_the_full_grace_before_unresponsive(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    clock.advance(3600)  # the coordinator was down for an hour
    again = Pool(cfg, observer, provider=provider, clock=clock)
    [n] = again.machines()
    provider.agents[n.instance_id].close()
    again.poll()
    assert n.state == MachineState.READY
    clock.advance(301)
    again.poll()
    assert n.state == MachineState.UNRESPONSIVE


def test_a_booting_slot_or_a_young_instance_is_not_an_orphan(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    from gpu4agents.core.models import Instance

    fp = cfg.fingerprint()[:6]
    prefix = pool.label_prefix()
    gate = threading.Event()
    provider.before_setup = gate.wait
    m = pool.open()  # s1 is renting
    try:
        provider.instances["x1"] = Instance(id="x1", provider="fake", label=f"{prefix}s1-{fp}", created_at=time.time() - 1000)
        provider.instances["x2"] = Instance(id="x2", provider="fake", label=f"{prefix}s9-{fp}", created_at=time.time())
        provider.instances["x3"] = Instance(id="x3", provider="fake", label=f"{prefix}s8-{fp}", created_at=time.time() - 1000)
        assert [i.id for i in pool.orphans()] == ["x3"]
    finally:
        gate.set()
        settle(pool, m)


# -- fix round 2 ----------------------------------------------------------------


def test_an_orphan_without_a_real_creation_time_is_aged_from_first_sight(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    from gpu4agents.core.models import Instance

    label = f"{pool.label_prefix()}s4-{cfg.fingerprint()[:6]}"
    provider.instances["x9"] = Instance(id="x9", provider="fake", label=label)

    def listed():
        provider.instances["x9"].created_at = clock()  # the mapper says "now", every time
        return [i.id for i in pool.orphans()]

    assert listed() == []
    clock.advance(60)
    assert listed() == []
    # The first sighting survives a restart of the pool.
    pool = Pool(cfg, observer, provider=provider, clock=clock)
    clock.advance(61)
    assert listed() == ["x9"]


def test_a_recovered_machine_s_restart_budget_resets(pool_parts):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    instance = provider.instances[m.instance_id]
    for _ in range(4):
        provider.stop(instance)
        pool.poll()
        assert m.state == MachineState.STOPPED
        provider.start(instance)
        settle(pool, m)  # back to checking: deadline and health check, then ready
        assert m.state == MachineState.READY


def test_a_pull_that_crashes_reports_every_job_lost(pool_parts, tmp_path, monkeypatch):
    pool, provider, observer, clock, cfg = pool_parts()
    m = settle(pool, pool.open())
    pool.dispatch(m, "q8-a1", write_result(tmp_path), {"run": "python3 main.py"}, {})
    wait_remote(pool, m, "q8-a1")

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(pool, "_pull_left_outputs", boom)
    result = pool.retire(m, "test")
    assert result.destroyed and result.lost == ["q8-a1"] and result.pulled == {}
    assert events(tmp_path, "pool_machine_retired")[0]["outputs_lost_ids"] == ["q8-a1"]
