"""A job's results are never deleted before someone downloaded them.

On the machine, the job runner turns a shutdown that would lose unsaved
results into one that keeps them. On this computer, the guard downloads them
before any automatic destroy, starting a stopped machine for it.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest
from fakes import FakeProvider
from test_jobs import runner, wait_done  # noqa: F401 - fixture
from test_watchdog import policy

from gpu4agents.core.beacon import WatchdogUnavailable
from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.engine import Engine
from gpu4agents.core.errors import ConfigError, StateError
from gpu4agents.core.job import JobRecords
from gpu4agents.core.state import utc_now

WRITES_A_RESULT = (
    "python3 -c \"import os; open(os.path.join(os.environ['JAMBU_OUTPUT_DIR'], 'model.txt'), 'w')"
    ".write('trained')\""
)
# Vast.ai's shape: a stop keeps the disk but still bills it, so idle destroys.
BILLING_STOP = policy(idle_timeout_s=2.0, stop_eliminates_billing=False)


# -- on the machine -------------------------------------------------------------


def test_unsaved_results_turn_an_idle_destroy_into_a_stop(runner):  # noqa: F811
    agent, client = runner(BILLING_STOP, interval_s=0.5, stop_keeps_disk=True)
    client.start("train", {"command": WRITES_A_RESULT})
    assert wait_done(client, "train")["outputs_unsaved"] is True

    marker = agent.wait_for_marker(timeout=20)
    assert marker and marker["action"] == "stop", "the disk holding the results must survive"
    assert agent.records("outputs_kept")[-1]["job_ids"] == ["train"]


def test_downloaded_results_let_the_machine_be_destroyed(runner, tmp_path):  # noqa: F811
    agent, client = runner(
        replace(BILLING_STOP, idle_timeout_s=4.0), interval_s=0.5, stop_keeps_disk=True
    )
    client.start("train", {"command": WRITES_A_RESULT})
    wait_done(client, "train")
    assert client.download_outputs("train", tmp_path / "back") == ["model.txt"]
    assert client.info("train")["outputs_unsaved"] is False
    assert [j["outputs_unsaved"] for j in client.list()] == [False]

    marker = agent.wait_for_marker(timeout=20)
    assert marker and marker["action"] == "destroy"
    assert "outputs_kept" not in agent.event_names()


def test_a_job_without_results_does_not_hold_anything(runner):  # noqa: F811
    agent, client = runner(BILLING_STOP, interval_s=0.5, stop_keeps_disk=True)
    client.start("noop", {"command": "true"})
    assert wait_done(client, "noop")["outputs_unsaved"] is False
    assert agent.wait_for_marker(timeout=20)["action"] == "destroy"


def test_where_a_stop_would_wipe_the_disk_an_idle_machine_waits_for_the_download(runner):  # noqa: F811
    # No persistent disk (Lambda, a RunPod pod without a volume): keep it up a while.
    agent, client = runner(
        BILLING_STOP, interval_s=0.5, env={"JAMBU_OUTPUTS_HOLD_S": "4"}
    )
    client.start("train", {"command": WRITES_A_RESULT})
    wait_done(client, "train")
    time.sleep(3.5)
    assert not agent.marker.exists(), "results nobody downloaded must hold the machine"
    assert "outputs_hold" in agent.event_names()

    # The hold is bounded: a forgotten machine still shuts down.
    marker = agent.wait_for_marker(timeout=20)
    assert marker and marker["action"] == "destroy"
    assert "outputs_lost" in agent.event_names()


# -- on this computer ---------------------------------------------------------------


class FakeJobs:
    """The machine's job runner as the engine sees it."""

    def __init__(self, job_id: str, fail: bool = False) -> None:
        self.job_id = job_id
        self.fail = fail
        self.unsaved = True

    def list(self):
        return [{"id": self.job_id, "state": "succeeded", "outputs_unsaved": self.unsaved}]

    def download_outputs(self, job_id, dest):
        if self.fail:
            raise WatchdogUnavailable("connection reset")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.txt").write_text("trained")
        self.unsaved = False
        return ["model.txt"]

    def info(self, job_id):
        return {"id": job_id, "state": "succeeded", "outputs_unsaved": self.unsaved}


class Pinging:
    """A watchdog that answers pings and reports nothing else."""

    def ping(self):
        return True

    def try_state(self):
        return None

    def events(self, limit: int = 100):
        return []


def vast_like(write_config, monkeypatch, fail=False, **lifecycle):
    config = write_config(lifecycle={"stopped_ttl": "1h", **lifecycle})
    provider = FakeProvider(config, CredentialResolver(config.project_dir))
    provider.capabilities = replace(
        provider.capabilities, persistent_disk=True, stop_eliminates_billing=False
    )
    engine = Engine(config, provider=provider)
    state = engine.setup(wait=False)
    JobRecords(engine.store.dir).save(
        {"id": "train", "instance_id": state.instance_id, "submitted_at": utc_now(), "state": "running"}
    )
    jobs = FakeJobs("train", fail=fail)
    from gpu4agents.core.models import InstanceState

    monkeypatch.setattr(
        engine,
        "_reachable_jobs",
        lambda s: jobs if InstanceState(s.state).is_billable_compute else None,
    )
    monkeypatch.setattr(engine, "beacon", lambda s: Pinging())
    return engine, state, jobs


def test_a_machine_stopped_to_keep_results_is_emptied_and_destroyed(write_config, monkeypatch):
    engine, state, jobs = vast_like(write_config, monkeypatch)
    engine.stop()  # what the watchdog did instead of destroying

    decision = engine.guard_tick()
    assert decision["action"] == "destroy" and decision["reason"] == "job_outputs_saved"
    calls = engine.provider.calls
    assert calls.index(("start", state.instance_id)) < calls.index(("destroy", state.instance_id))
    landed = engine.config.project_dir / "gpu-outputs" / "train" / "model.txt"
    assert landed.read_text() == "trained"
    assert JobRecords(engine.store.dir).load("train")["outputs_unsaved"] is False


def test_results_that_cannot_be_fetched_keep_the_machine(write_config, monkeypatch):
    engine, state, jobs = vast_like(write_config, monkeypatch, fail=True)
    engine.stop()
    with engine.store.transaction() as stored:
        stored.stopped_since = utc_now() - 7200  # past stopped_ttl

    decision = engine.guard_tick()
    assert decision["action"] == "stop", "started to fetch them, then stopped again"
    assert ("destroy", state.instance_id) not in engine.provider.calls
    assert engine.store.load().instance_id == state.instance_id
    assert "job_outputs_kept" in [r["event"] for r in engine.events.tail(limit=50)]

    # The next guard does not restart it again right away.
    starts = engine.provider.calls.count(("start", state.instance_id))
    assert engine.guard_tick()["action"] == "none"
    assert engine.provider.calls.count(("start", state.instance_id)) == starts


def test_keep_disk_leaves_a_stopped_machine_with_results_alone_until_stopped_ttl(
    write_config, monkeypatch
):
    engine, state, jobs = vast_like(write_config, monkeypatch, keep_disk=True)
    engine.stop()
    assert engine.guard_tick()["action"] == "none"
    assert ("start", state.instance_id) not in engine.provider.calls


def test_destroy_downloads_results_first(write_config, monkeypatch):
    engine, state, jobs = vast_like(write_config, monkeypatch)
    engine.destroy()
    assert (engine.config.project_dir / "gpu-outputs" / "train" / "model.txt").exists()
    assert engine.store.load().instance_id is None


def test_destroy_refuses_to_delete_results_it_could_not_fetch(write_config, monkeypatch):
    engine, state, jobs = vast_like(write_config, monkeypatch, fail=True)
    with pytest.raises(StateError, match="--discard-outputs"):
        engine.destroy()
    assert engine.store.load().instance_id == state.instance_id

    engine.destroy(discard_outputs=True)
    assert engine.store.load().instance_id is None


def test_job_pull_starts_a_stopped_machine_and_puts_it_back(write_config, monkeypatch):
    engine, state, jobs = vast_like(write_config, monkeypatch)
    monkeypatch.setattr(engine, "_job_client", lambda start: (engine.store.load(), jobs))
    engine.stop()

    pulled = engine.job_pull("train")
    assert pulled["files"] == ["model.txt"]
    # Nothing left on it, and a stopped Vast.ai machine bills: it is destroyed.
    assert engine.store.load().instance_id is None


def test_status_lists_results_only_on_the_machine(write_config, monkeypatch):
    engine, state, jobs = vast_like(write_config, monkeypatch)
    assert engine.status()["job_outputs_unsaved"] == ["train"]


def test_a_job_longer_than_max_lifetime_is_refused_before_anything_is_rented(write_config, tmp_path):
    config = write_config(lifecycle={"max_lifetime": "2h", "idle_timeout": "15m"})
    engine = Engine(config, provider=FakeProvider(config, CredentialResolver(config.project_dir)))
    code = tmp_path / "code"
    code.mkdir()
    (code / "gpujob.yaml").write_text("run: python3 train.py\ntimeout: 10h\n")
    with pytest.raises(ConfigError, match="max_lifetime"):
        engine.job_submit(code)
    assert engine.provider.calls == []


def test_without_a_lasting_disk_the_guard_fetches_results_while_the_machine_waits(
    write_config, monkeypatch
):
    engine, state, jobs = vast_like(write_config, monkeypatch)
    engine.provider.capabilities = replace(engine.provider.capabilities, persistent_disk=False)
    decision = engine.guard_tick()
    assert decision["action"] == "none"
    assert (engine.config.project_dir / "gpu-outputs" / "train" / "model.txt").exists()
    assert engine.status()["job_outputs_unsaved"] == []
