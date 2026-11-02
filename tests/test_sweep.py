"""Account-wide control: list_instances, the project registry, `gpu ls`, `gpu sweep`, the timer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import respx
import yaml
from conftest import BASE_CONFIG, _deep_merge
from fakes import FakeProvider
from typer.testing import CliRunner

import gpu4agents
from gpu4agents.cli import _common
from gpu4agents.cli import account as account_cli
from gpu4agents.cli import timer as timer_mod
from gpu4agents.cli.main import app
from gpu4agents.core import sweep as sweep_mod
from gpu4agents.core.config import load_config
from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.engine import Engine
from gpu4agents.core.errors import ProviderError, UnsupportedCapabilityError
from gpu4agents.core.events import EventLog
from gpu4agents.core.models import Instance, InstanceState, ProviderCapabilities
from gpu4agents.core.state import utc_now
from gpu4agents.providers.base import GPUProvider
from gpu4agents.providers.vast.config import API_BASE
from gpu4agents.providers.vast.provider import VastProvider

runner = CliRunner()
HOUR = 3600.0


@pytest.fixture
def fake(config):
    return FakeProvider(config, CredentialResolver(config.project_dir))


def add(provider, instance_id, state, age_s, label="jambu-model-abc123"):
    provider.instances[instance_id] = Instance(
        id=instance_id,
        provider=provider.name,
        state=state,
        created_at=utc_now() - age_s,
        gpu_name="RTX 4090",
        gpu_count=1,
        hourly_cost_usd=0.4,
        label=label,
    )
    return provider.instances[instance_id]


def run_sweep(provider, apply=True, tracked=None, registry=None, now=None, events=None, **kw):
    return sweep_mod.sweep(
        provider,
        orphan_max_age_s=kw.get("orphan_max_age_s", 24 * HOUR),
        stopped_ttl_s=kw.get("stopped_ttl_s", 24 * HOUR),
        apply=apply,
        tracked=tracked,
        registry=registry or sweep_mod.SweepRegistry(),
        events=events,
        now=now,
    )


def write_project(directory: Path, **overrides):
    """A jambu.yaml of its own in ``directory`` (a second, third ... project)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "jambu.yaml"
    data = _deep_merge(BASE_CONFIG, overrides)
    path.write_text(yaml.safe_dump({"gpu_runtime": data}, sort_keys=False))
    return load_config(path)


# -- provider contract ---------------------------------------------------------


def test_list_instances_is_an_optional_capability(config):
    assert ProviderCapabilities().list_instances is False

    class Minimal(FakeProvider):
        capabilities = ProviderCapabilities(stop=True)

        def list_instances(self):
            return GPUProvider.list_instances(self)

    with pytest.raises(UnsupportedCapabilityError):
        Minimal(config, CredentialResolver()).list_instances()
    with pytest.raises(UnsupportedCapabilityError):
        sweep_mod.list_jambu_instances(Minimal(config, CredentialResolver()))


@respx.mock
def test_vast_lists_only_jambu_labelled_instances(config, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    provider = VastProvider(config, CredentialResolver(config.project_dir))
    assert provider.capabilities.list_instances is True
    route = respx.get(f"{API_BASE}/instances/").mock(
        return_value=httpx.Response(
            200,
            json={
                "instances": [
                    {"id": 1, "label": "jambu-Qwythos-9B-v2-abc123", "actual_status": "running",
                     "dph_total": 0.35, "gpu_name": "RTX 4090", "num_gpus": 1,
                     "start_date": 1_700_000_000.0},
                    {"id": 2, "label": "my-own-training-run", "actual_status": "running"},
                    {"id": 3, "label": None, "actual_status": "stopped"},
                    {"id": 4, "label": "jambu-other-def456", "actual_status": "exited"},
                ]
            },
        )
    )
    found = provider.list_instances()
    assert route.calls[0].request.url.params["owner"] == "me"
    assert [i.id for i in found] == ["1", "4"]
    assert found[0].state is InstanceState.READY
    assert found[0].hourly_cost_usd == 0.35
    assert found[1].state is InstanceState.STOPPED


def test_the_engine_labels_instances_so_the_sweep_can_find_them(config, fake):
    engine = Engine(config, provider=fake)
    state = engine.setup(wait=False)
    assert [i.id for i in sweep_mod.list_jambu_instances(fake)] == [state.instance_id]


# -- orphan rules ---------------------------------------------------------------


def test_a_running_orphan_past_orphan_max_age_is_stopped_never_destroyed(fake):
    add(fake, "old", InstanceState.READY, 25 * HOUR)
    add(fake, "young", InstanceState.READY, 23 * HOUR)

    result = run_sweep(fake)
    assert [(a.instance.id, a.action, a.reason) for a in result.actions] == [
        ("old", "stop", "orphan_max_age")
    ]
    assert ("stop", "old") in fake.calls
    assert not [c for c in fake.calls if c[0] == "destroy"]
    assert fake.instances["old"].state is InstanceState.STOPPED
    assert fake.instances["young"].state is InstanceState.READY


def test_a_running_orphan_is_not_destroyed_even_when_stop_is_unsupported(fake):
    fake.capabilities = ProviderCapabilities(destroy=True, list_instances=True)
    add(fake, "old", InstanceState.READY, 99 * HOUR)
    result = run_sweep(fake)
    assert [(a.action, a.reason) for a in result.actions] == [("none", "stop_unsupported")]
    assert not [c for c in fake.calls if c[0] in ("stop", "destroy")]


def test_a_running_orphan_without_a_creation_time_is_left_alone(fake):
    add(fake, "ageless", InstanceState.READY, 0).created_at = 0.0
    assert run_sweep(fake).actions == []


def test_tracked_instances_are_left_entirely_to_their_project(fake, jambu_home):
    """Bug: the sweep judged other projects' instances by one project's rules."""
    add(fake, "busy", InstanceState.READY, 100 * HOUR)
    add(fake, "parked", InstanceState.STOPPED, 100 * HOUR)
    t0 = utc_now()
    run_sweep(fake, tracked={"busy", "parked"}, now=t0)
    result = run_sweep(fake, tracked={"busy", "parked"}, now=t0 + 100 * HOUR)
    assert result.actions == [] and result.tracked == 2
    assert not [c for c in fake.calls if c[0] in ("stop", "destroy")]
    assert json.loads((jambu_home / "sweep.json").read_text())["stopped"] == {}


def test_non_jambu_instances_are_never_touched(fake):
    add(fake, "theirs", InstanceState.READY, 100 * HOUR, label="someone-elses-job")
    add(fake, "unlabelled", InstanceState.STOPPED, 100 * HOUR, label="")

    class Sloppy(FakeProvider):
        def list_instances(self):  # an adapter that forgot to filter
            return list(self.instances.values())

    sloppy = Sloppy(fake.config, CredentialResolver())
    sloppy.instances = fake.instances
    result = run_sweep(sloppy, now=utc_now() + 100 * HOUR)
    assert result.seen == 0 and result.actions == []
    assert not [c for c in sloppy.calls if c[0] in ("stop", "destroy")]


def test_a_stopped_orphan_is_destroyed_once_seen_stopped_past_stopped_ttl(fake, jambu_home):
    add(fake, "parked", InstanceState.STOPPED, 2 * HOUR)
    t0 = utc_now()

    first = run_sweep(fake, now=t0)
    assert first.actions == []
    saved = json.loads((jambu_home / "sweep.json").read_text())
    assert "fake:parked" in saved["stopped"]

    # The first sighting is kept across runs; a day later it goes.
    assert run_sweep(fake, now=t0 + 12 * HOUR).actions == []
    later = run_sweep(fake, now=t0 + 25 * HOUR)
    assert [(a.action, a.reason) for a in later.actions] == [("destroy", "stopped_ttl")]
    assert "parked" not in fake.instances
    assert "fake:parked" not in json.loads((jambu_home / "sweep.json").read_text())["stopped"]


def test_a_stopped_orphan_counts_from_the_sweeps_own_first_sighting(fake):
    """Bug: a project's stale stopped_since was copied in, destroying a fresh stop.
    Only the sweep's own sighting counts, never the instance's age."""
    add(fake, "parked", InstanceState.STOPPED, 72 * HOUR)
    assert run_sweep(fake).actions == []
    assert "parked" in fake.instances


def test_registry_entries_go_when_the_instance_runs_again_or_disappears(fake, jambu_home):
    add(fake, "a", InstanceState.STOPPED, HOUR)
    add(fake, "b", InstanceState.STOPPED, HOUR)
    run_sweep(fake)
    assert set(json.loads((jambu_home / "sweep.json").read_text())["stopped"]) == {
        "fake:a",
        "fake:b",
    }

    fake.instances["a"].state = InstanceState.READY
    del fake.instances["b"]
    run_sweep(fake)
    assert json.loads((jambu_home / "sweep.json").read_text())["stopped"] == {}


def test_a_dry_run_changes_nothing_but_still_records_sightings(fake, jambu_home):
    add(fake, "old", InstanceState.READY, 30 * HOUR)
    add(fake, "parked", InstanceState.STOPPED, HOUR)
    result = run_sweep(fake, apply=False)
    assert result.actions[0].action == "stop" and not result.actions[0].applied
    assert not [c for c in fake.calls if c[0] in ("stop", "destroy")]
    assert "fake:parked" in json.loads((jambu_home / "sweep.json").read_text())["stopped"]


def test_null_thresholds_disable_the_rules(fake):
    add(fake, "old", InstanceState.READY, 99 * HOUR)
    add(fake, "parked", InstanceState.STOPPED, HOUR)
    result = run_sweep(fake, orphan_max_age_s=None, stopped_ttl_s=None, now=utc_now() + 99 * HOUR)
    assert result.actions == []


def test_every_action_emits_an_event_with_its_reason(fake, tmp_path):
    add(fake, "old", InstanceState.READY, 30 * HOUR)
    log = EventLog(tmp_path / "logs")
    run_sweep(fake, events=[log])
    records = log.tail()
    sweep_events = [r for r in records if r["event"] == "sweep_action"]
    assert sweep_events[0]["reason"] == "orphan_max_age"
    assert sweep_events[0]["applied"] is True
    stopped = [r for r in records if r["event"] == "instance_stopped"]
    assert stopped[0]["reason"] == "orphan_max_age"
    assert stopped[0]["source"] == "sweep"


def test_a_provider_error_on_one_instance_does_not_stop_the_sweep(fake):
    add(fake, "a", InstanceState.READY, 30 * HOUR)
    add(fake, "b", InstanceState.READY, 30 * HOUR)
    real_stop = fake.stop

    def flaky(instance):
        if instance.id == "a":
            raise ProviderError("api blip")
        real_stop(instance)

    fake.stop = flaky
    result = run_sweep(fake)
    by_id = {a.instance.id: a for a in result.actions}
    assert by_id["a"].error == "api blip" and not by_id["a"].applied
    assert by_id["b"].applied and fake.instances["b"].state is InstanceState.STOPPED


def test_a_corrupt_registry_does_not_block_the_sweep(fake, jambu_home):
    jambu_home.mkdir(parents=True)
    (jambu_home / "sweep.json").write_text("{not json")
    add(fake, "old", InstanceState.READY, 30 * HOUR)
    assert run_sweep(fake).actions[0].applied


# -- project registry -------------------------------------------------------------


def test_the_project_registry_keeps_absolute_paths_once(tmp_path, jambu_home, monkeypatch):
    registry = sweep_mod.ProjectRegistry()
    assert registry.path == jambu_home / "projects.json"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "jambu.yaml").write_text("x")
    registry.register(Path("p/jambu.yaml"))
    registry.register(tmp_path / "p" / "jambu.yaml")
    assert registry.paths() == [(tmp_path / "p" / "jambu.yaml").resolve()]
    registry.remove(registry.paths())
    assert registry.paths() == []


def test_a_corrupt_project_registry_reads_as_empty(jambu_home):
    jambu_home.mkdir(parents=True)
    (jambu_home / "projects.json").write_text("[oops")
    assert sweep_mod.ProjectRegistry().paths() == []


def test_setup_registers_the_project(config, monkeypatch, jambu_home):
    monkeypatch.chdir(config.project_dir)
    provider = FakeProvider(config, CredentialResolver())
    monkeypatch.setattr(
        _common.Context, "engine", lambda self: Engine(self.config(), provider=provider)
    )
    result = runner.invoke(app, ["--json", "setup", "--no-wait"])
    assert result.exit_code == 0, result.output
    assert sweep_mod.ProjectRegistry().paths() == [config.source_path.resolve()]


# -- `gpu sweep`: every project by its own policy, then the orphans -------------


@pytest.fixture
def cli_fake(monkeypatch, tmp_path):
    """Every CLI invocation in a test talks to the same in-memory account."""
    holder = {}

    def _make(config, credentials, name):
        if "provider" not in holder:
            holder["provider"] = FakeProvider(config, credentials)
        return holder["provider"]

    monkeypatch.setattr(account_cli, "make_provider", _make)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    return holder


def _seed(cli_fake, config):
    provider = FakeProvider(config, CredentialResolver())
    cli_fake["provider"] = provider
    return provider


def _project(provider, directory, register=True, **overrides):
    """Provision a project's instance on the shared fake account."""
    config = write_project(directory, **overrides)
    engine = Engine(config, provider=provider)
    state = engine.setup(wait=False)
    if register:
        sweep_mod.ProjectRegistry().register(config.source_path)
    return engine, state


def _sweep_json(*args):
    result = runner.invoke(app, ["--json", "sweep", *args])
    return result, json.loads(result.stdout)


def test_another_projects_restarted_and_locked_instance_is_not_destroyed(
    config, cli_fake, monkeypatch, tmp_path
):
    """Bug: `sweep --apply` in project A destroyed project B's instance, in use
    and locked, because its provider creation time was past A's max_lifetime."""
    monkeypatch.chdir(config.project_dir)
    provider = _seed(cli_fake, config)
    other, state = _project(provider, tmp_path / "b")
    provider.instances[state.instance_id].created_at = utc_now() - 7 * HOUR
    with other.store.transaction() as stored:
        stored.session_started_at = utc_now() - HOUR  # restarted an hour ago
        stored.locked, stored.locked_until = True, utc_now() + 12 * HOUR
        stored.last_activity, stored.starting = utc_now(), False

    result, payload = _sweep_json("--apply")
    assert result.exit_code == 0, result.output
    assert not [c for c in provider.calls if c[0] in ("stop", "destroy")]
    assert payload["actions"] == [] and payload["tracked"] == 1
    guarded = {p["config"]: p for p in payload["projects"]}
    assert guarded[str(other.config.source_path)]["decision"]["action"] == "none"


def test_each_project_is_judged_by_its_own_max_lifetime(cli_fake, monkeypatch, tmp_path):
    nowhere = tmp_path / "nowhere"
    nowhere.mkdir()
    monkeypatch.chdir(nowhere)  # the timer runs from ~, not from a project
    provider = _seed(cli_fake, write_project(tmp_path / "seed"))
    long, long_state = _project(provider, tmp_path / "long", lifecycle={"max_lifetime": "12h"})
    short, short_state = _project(provider, tmp_path / "short", lifecycle={"max_lifetime": "6h"})
    for engine in (long, short):
        with engine.store.transaction() as stored:
            stored.session_started_at = utc_now() - 9 * HOUR
            stored.last_activity, stored.starting = utc_now(), False

    result, payload = _sweep_json("--apply")
    assert result.exit_code == 0, result.output
    assert ("stop", short_state.instance_id) in provider.calls
    assert ("stop", long_state.instance_id) not in provider.calls
    decisions = {p["instance_id"]: p["decision"] for p in payload["projects"]}
    assert decisions[short_state.instance_id]["reason"] == "max_lifetime"
    assert decisions[long_state.instance_id]["action"] == "none"
    # The project's own guard logged it, in the project's own event log.
    assert any(r["event"] == "lifecycle_guard_triggered" for r in short.events.tail())


def test_a_project_whose_config_is_gone_is_unregistered(cli_fake, monkeypatch, tmp_path, config):
    monkeypatch.chdir(tmp_path)
    provider = _seed(cli_fake, config)
    engine, state = _project(provider, tmp_path / "gone")
    engine.config.source_path.unlink()

    result, payload = _sweep_json("--apply")
    assert result.exit_code == 0, result.output
    assert payload["unregistered"] == [str(engine.config.source_path)]
    assert sweep_mod.ProjectRegistry().paths() == []
    # Now an orphan, but a young running one: left alone.
    assert payload["tracked"] == 0 and payload["actions"] == []


def test_a_project_whose_config_no_longer_loads_still_owns_its_instance(
    cli_fake, monkeypatch, tmp_path, config
):
    monkeypatch.chdir(tmp_path)
    provider = _seed(cli_fake, config)
    engine, state = _project(provider, tmp_path / "broken")
    provider.instances[state.instance_id].created_at = utc_now() - 30 * HOUR
    engine.config.source_path.write_text("gpu_runtime: [not, a, mapping]\n")

    result, payload = _sweep_json("--apply")
    assert result.exit_code == 1
    assert "config does not load" in payload["projects"][0]["error"]
    assert payload["tracked"] == 1 and payload["actions"] == []
    assert not [c for c in provider.calls if c[0] in ("stop", "destroy")]
    assert sweep_mod.ProjectRegistry().paths() == [engine.config.source_path]


def test_sweep_guards_the_current_project_even_unregistered(config, cli_fake, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    provider = _seed(cli_fake, config)
    engine = Engine(config, provider=provider)
    state = engine.setup(wait=False)
    provider.instances[state.instance_id].created_at = utc_now() - 30 * HOUR
    with engine.store.transaction() as stored:
        stored.session_started_at = utc_now() - HOUR
        stored.last_activity, stored.starting = utc_now(), False

    result, payload = _sweep_json("--apply")
    assert result.exit_code == 0, result.output
    assert payload["tracked"] == 1 and payload["actions"] == []
    assert not [c for c in provider.calls if c[0] in ("stop", "destroy")]


def test_sweep_config_flag_guards_that_project_ad_hoc(cli_fake, monkeypatch, tmp_path, config):
    monkeypatch.chdir(tmp_path)
    provider = _seed(cli_fake, config)
    engine, state = _project(provider, tmp_path / "adhoc", register=False)
    with engine.store.transaction() as stored:
        stored.session_started_at = utc_now() - 7 * HOUR
        stored.last_activity, stored.starting = utc_now(), False

    result, payload = _sweep_json("--apply", "--config", str(engine.config.source_path))
    assert result.exit_code == 0, result.output
    assert payload["projects"][0]["decision"]["reason"] == "max_lifetime"
    assert ("stop", state.instance_id) in provider.calls
    assert sweep_mod.ProjectRegistry().paths() == []  # ad hoc: not registered


def test_sweep_is_a_dry_run_by_default(tmp_path, cli_fake, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    provider = _seed(cli_fake, config)
    add(provider, "old", InstanceState.READY, 30 * HOUR)

    result = runner.invoke(app, ["sweep"])
    assert result.exit_code == 0, result.output
    assert "would stop orphan old" in result.stdout
    assert "--apply" in result.stdout
    assert provider.instances["old"].state is InstanceState.READY

    result, payload = _sweep_json("--apply")
    assert result.exit_code == 0, result.output
    assert payload["applied"] is True
    assert payload["actions"][0]["reason"] == "orphan_max_age"
    assert provider.instances["old"].state is InstanceState.STOPPED


def test_sweep_flags_set_the_orphan_thresholds(tmp_path, cli_fake, monkeypatch, config, jambu_home):
    monkeypatch.chdir(tmp_path)
    provider = _seed(cli_fake, config)
    add(provider, "nine-hours", InstanceState.READY, 9 * HOUR)

    _, payload = _sweep_json()
    assert payload["orphan_max_age_s"] == 24 * HOUR and payload["stopped_ttl_s"] == 24 * HOUR
    assert payload["actions"] == []

    result, payload = _sweep_json("--apply", "--orphan-max-age", "6h")
    assert result.exit_code == 0, result.output
    assert payload["actions"][0]["instance_id"] == "nine-hours"
    events = (jambu_home / "logs" / "events.jsonl").read_text()
    assert "sweep_action" in events and "orphan_max_age" in events


def test_sweep_rejects_a_bad_duration_cleanly(config, cli_fake, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    _seed(cli_fake, config)
    result = runner.invoke(app, ["sweep", "--stopped-ttl", "soon"])
    assert result.exit_code == 2
    assert "--stopped-ttl" in result.output


def test_ls_names_the_project_owning_each_instance(config, cli_fake, monkeypatch, tmp_path):
    monkeypatch.chdir(config.project_dir)
    provider = _seed(cli_fake, config)
    mine = Engine(config, provider=provider).setup(wait=False)
    other, other_state = _project(provider, tmp_path / "other")
    add(provider, "elsewhere", InstanceState.STOPPED, 3 * HOUR, label="jambu-other-000000")
    add(provider, "theirs", InstanceState.READY, HOUR, label="not-ours")

    result = runner.invoke(app, ["--json", "ls"])
    assert result.exit_code == 0, result.output
    rows = {r["instance_id"]: r for r in json.loads(result.stdout)["instances"]}
    assert set(rows) == {mine.instance_id, other_state.instance_id, "elsewhere"}
    assert rows[mine.instance_id]["project"] == str(config.source_path)
    assert rows[other_state.instance_id]["project"] == str(other.config.source_path)
    assert rows["elsewhere"]["tracked"] is False and rows["elsewhere"]["project"] is None
    assert rows["elsewhere"]["state"] == "stopped"
    assert rows["elsewhere"]["hourly_cost_usd"] == 0.4
    assert rows["elsewhere"]["age_seconds"] == pytest.approx(3 * HOUR, abs=5)

    table = runner.invoke(app, ["ls"])
    assert table.exit_code == 0
    assert "elsewhere" in table.stdout and "theirs" not in table.stdout


def test_ls_works_without_a_project(tmp_path, cli_fake, monkeypatch, config):
    empty = tmp_path / "nowhere"
    empty.mkdir()
    monkeypatch.chdir(empty)
    provider = _seed(cli_fake, config)
    add(provider, "x", InstanceState.READY, HOUR)
    result = runner.invoke(app, ["--json", "ls"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["instances"][0]["tracked"] is False


# -- systemd timer -------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    return home


@pytest.fixture
def credentials_file(home, jambu_home):
    path = jambu_home / "credentials"
    path.parent.mkdir(parents=True)
    path.write_text("VAST_API_KEY=from-file\n")
    return path


def test_install_timer_writes_one_unit_independent_of_the_project(
    config, home, credentials_file, monkeypatch, jambu_home
):
    monkeypatch.chdir(config.project_dir)
    result = runner.invoke(app, ["sweep", "--install-timer", "--every", "10m"])
    assert result.exit_code == 0, result.output

    unit_dir = home / ".config" / "systemd" / "user"
    service = (unit_dir / "jambu-gpu-sweep.service").read_text()
    timer = (unit_dir / "jambu-gpu-sweep.timer").read_text()
    execs = [line for line in service.splitlines() if line.startswith("ExecStart")]
    assert execs == [f"ExecStart={sys.executable} -m gpu4agents sweep --apply"]  # no "-"
    assert "WorkingDirectory=~" in service
    assert "TimeoutStartSec=10min" in service
    assert f'Environment="GPU_HOME={jambu_home}"' in service
    assert "Type=oneshot" in service
    assert "OnUnitActiveSec=600s" in timer
    assert "WantedBy=timers.target" in timer
    assert sweep_mod.ProjectRegistry().paths() == [config.source_path.resolve()]
    printed = " ".join(result.stdout.split())  # rich wraps long lines
    assert "systemctl --user enable --now jambu-gpu-sweep.timer" in printed


def test_install_timer_from_two_projects_keeps_one_unit_and_both_registered(
    write_config, home, credentials_file, monkeypatch, tmp_path
):
    config = write_config()
    monkeypatch.chdir(config.project_dir)
    runner.invoke(app, ["sweep", "--install-timer"])
    other = write_project(tmp_path / "other project")
    monkeypatch.chdir(other.project_dir)
    result = runner.invoke(app, ["--json", "sweep", "--install-timer"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)
    assert set(payload["projects"]) == {str(config.source_path), str(other.source_path)}
    service = Path(payload["service"]).read_text()
    assert len([line for line in service.splitlines() if line.startswith("ExecStart")]) == 1
    assert not [line for line in service.splitlines() if line.endswith(" guard")]


def test_install_timer_migrates_old_per_project_guard_lines(
    home, credentials_file, monkeypatch, tmp_path
):
    unit_dir = home / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    legacy = write_project(tmp_path / "legacy dir")
    (unit_dir / "jambu-gpu-sweep.service").write_text(
        "[Service]\nExecStart=-/usr/bin/python3 -m gpu4agents sweep --apply\n"
        f'ExecStart=-/usr/bin/python3 -m gpu4agents --config "{legacy.source_path}" guard\n'
    )
    empty = tmp_path / "nowhere"
    empty.mkdir()
    monkeypatch.chdir(empty)
    result = runner.invoke(app, ["--json", "sweep", "--install-timer"])
    assert result.exit_code == 0, result.output
    assert sweep_mod.ProjectRegistry().paths() == [legacy.source_path]
    rewritten = (unit_dir / "jambu-gpu-sweep.service").read_text()
    assert not [line for line in rewritten.splitlines() if line.endswith(" guard")]


def test_install_timer_refuses_a_key_only_in_the_shell_environment(config, home, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    monkeypatch.setenv("VAST_API_KEY", "only-in-my-shell")
    result = runner.invoke(app, ["sweep", "--install-timer"])
    assert result.exit_code == 2
    message = " ".join(result.output.split())
    assert "~/.gpu/credentials" in message and "VAST_API_KEY" in message
    assert not (home / ".config/systemd/user/jambu-gpu-sweep.service").exists()


def test_install_timer_accepts_the_key_in_the_project_env(config, home, monkeypatch):
    (config.project_dir / ".env").write_text("VAST_API_KEY=in-the-project\n")
    monkeypatch.chdir(config.project_dir)
    result = runner.invoke(app, ["sweep", "--install-timer"])
    assert result.exit_code == 0, result.output
    # ... and the orphan sweep, running from ~, finds it through the registry.
    assert account_cli.account_credentials(home).get("VAST_API_KEY") == "in-the-project"


def test_unit_values_escape_percent_and_dollar_and_quote_spaces():
    service, _ = timer_mod.render_units(
        every_s=300, python="/opt/my env/100%/py$thon", jambu_home="/data/50% \"x\""
    )
    assert 'ExecStart="/opt/my env/100%%/py$$thon" -m gpu4agents sweep --apply' in service
    assert 'Environment="GPU_HOME=/data/50%% \\"x\\""' in service
    assert "Environment" not in timer_mod.render_units(every_s=300, python="/p")[0]


def test_install_timer_without_a_project_only_sweeps(tmp_path, home, credentials_file, monkeypatch):
    empty = tmp_path / "nowhere"
    empty.mkdir()
    monkeypatch.chdir(empty)
    result = runner.invoke(app, ["--json", "sweep", "--install-timer"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["project_config"] is None and payload["projects"] == []
    assert "sweep --apply" in Path(payload["service"]).read_text()


def test_uninstall_timer_removes_the_units(config, home, credentials_file, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    runner.invoke(app, ["sweep", "--install-timer"])
    result = runner.invoke(app, ["--json", "sweep", "--uninstall-timer"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload["removed"]) == 2
    assert "disable --now" in payload["disable"]
    assert not (home / ".config/systemd/user/jambu-gpu-sweep.timer").exists()
    assert runner.invoke(app, ["sweep", "--uninstall-timer"]).exit_code == 0


def test_the_module_entrypoint_the_timer_runs_works():
    repo = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-m", "gpu4agents", "version"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == gpu4agents.__version__
