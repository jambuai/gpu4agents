"""Fixes from the first real, filmed use of 0.12.0 (docs/handoff/2026-11-11-video-validation-fixes.md).

Each section names the item it covers (G1-G5). The scenarios are the ones
observed in that session, not invented ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import BASE_CONFIG, _deep_merge
from fakes import FakeProvider
from typer.testing import CliRunner

import yaml

from gpu4agents.cli import _common
from gpu4agents.cli import account as account_cli
from gpu4agents.cli import timer as timer_mod
from gpu4agents.cli.main import app
from gpu4agents.core import sweep as sweep_mod
from gpu4agents.core.config import load_config
from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.engine import Engine
from gpu4agents.core.models import ProviderCapabilities
from gpu4agents.core.state import utc_now

runner = CliRunner()
HOUR = 3600.0


class BillingStopProvider(FakeProvider):
    """Like Vast.ai: a stopped machine still bills its disk."""

    capabilities = ProviderCapabilities(
        **{**FakeProvider.capabilities.__dict__, "stop_eliminates_billing": False, "persistent_disk": True}
    )

    def setup(self, spec):
        instance = super().setup(spec)
        instance.stopped_hourly_cost_usd = 0.0219  # 80 GB at $0.20/GB-month
        return instance


def billing_engine(config):
    return Engine(config, provider=BillingStopProvider(config, CredentialResolver(config.project_dir)))


def write_project(directory: Path, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gpu.yaml"
    path.write_text(yaml.safe_dump({"gpu_runtime": _deep_merge(BASE_CONFIG, overrides)}, sort_keys=False))
    return load_config(path)


@pytest.fixture
def no_timer(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(timer_mod, "_platform", lambda: "linux")
    return home


# -- G1: a manual stop that keeps billing ------------------------------------------


def test_stop_where_a_stopped_machine_bills_says_so_and_names_the_bill(config, no_timer):
    engine = billing_engine(config)
    engine.setup(wait=False)
    result = engine.stop()
    assert result["action"] == "stop"
    assert result["still_billing"] is True
    assert result["stopped_cost_usd_per_day"] == pytest.approx(0.53, abs=0.01)
    assert result["sweep_timer_installed"] is False
    warning = result["warning"]
    assert "still billing ~$0.53/day" in warning
    assert "gpu sweep --install-timer" in warning and "gpu destroy" in warning


def test_stop_where_stopping_ends_the_bill_carries_no_warning(config, no_timer):
    engine = Engine(config, provider=FakeProvider(config, CredentialResolver(config.project_dir)))
    engine.setup(wait=False)
    result = engine.stop()
    assert result["still_billing"] is False
    assert "warning" not in result


def test_stop_with_the_timer_installed_says_when_it_goes(config, no_timer):
    service, timer = timer_mod.unit_paths()
    timer.parent.mkdir(parents=True)
    timer.write_text("[Timer]\n")
    wants = timer.parent / "timers.target.wants"
    wants.mkdir()
    (wants / timer.name).symlink_to(timer)
    engine = billing_engine(config)
    engine.setup(wait=False)
    result = engine.stop()
    assert result["sweep_timer_installed"] is True
    assert "destroyed after stopped_ttl (1d)" in result["warning"]


def test_a_written_but_disabled_timer_does_not_count(config, no_timer):
    service, timer = timer_mod.unit_paths()
    timer.parent.mkdir(parents=True)
    timer.write_text("[Timer]\n")
    assert timer_mod.is_installed() is False


def test_stop_with_stopped_ttl_off_says_it_bills_until_destroyed(write_config, no_timer):
    config = write_config(lifecycle={"stopped_ttl": None, "keep_disk": True})
    engine = billing_engine(config)
    engine.setup(wait=False)
    assert "until `gpu destroy`" in engine.stop()["warning"]


def test_cli_stop_prints_the_bill(config, no_timer, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    engine = billing_engine(config)
    engine.setup(wait=False)
    monkeypatch.setattr(_common.Context, "engine", lambda self: engine)
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0, result.output
    assert "still billing" in " ".join(result.output.split())


def test_acceptance_manual_stop_then_only_the_timer_runs(write_config, no_timer, monkeypatch, tmp_path):
    """Take 2: the agent ran `gpu stop` and nothing else ever ran. With the
    sweep timer the user agreed to at install, the machine is destroyed once
    stopped_ttl passes, by the timer's `gpu sweep --apply` alone."""
    config = write_config(lifecycle={"stopped_ttl": "24h"})
    provider = BillingStopProvider(config, CredentialResolver(config.project_dir))
    engine = Engine(config, provider=provider)
    engine.setup(wait=False)
    sweep_mod.ProjectRegistry().register(config.source_path)
    engine.stop()

    # The timer's run: no project directory, no shell, just `gpu sweep --apply`.
    def guard(cfg):
        return Engine(cfg, provider=provider)

    guards, _ = sweep_mod.guard_projects(sweep_mod.ProjectRegistry().paths(), guard, apply=True)
    assert guards[0].decision["action"] == "none"  # within stopped_ttl

    # A day passes with nothing running.
    with engine.store.transaction() as stored:
        for name in ("stopped_since", "session_started_at", "last_activity", "created_at"):
            setattr(stored, name, getattr(stored, name) - 25 * HOUR)
        stored.watchdog.last_contact = None
    for instance in provider.instances.values():
        instance.created_at -= 25 * HOUR
    guards, _ = sweep_mod.guard_projects(sweep_mod.ProjectRegistry().paths(), guard, apply=True)
    assert guards[0].decision["action"] == "destroy"
    assert guards[0].decision["reason"] == "stopped_ttl"
    assert not provider.instances


def test_every_command_warns_about_a_stopped_machine_still_billing(config, no_timer, monkeypatch, tmp_path):
    """A later session, in any directory, hears about the bill the last one left."""
    engine = billing_engine(config)
    engine.setup(wait=False)
    sweep_mod.ProjectRegistry().register(config.source_path)
    engine.stop()

    found = sweep_mod.stopped_and_billing()
    assert [f["instance_id"] for f in found] == [engine.store.load().instance_id]
    assert found[0]["stopped_cost_usd_per_day"] == pytest.approx(0.53, abs=0.01)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "still billing" in " ".join(result.stderr.split())
    assert str(config.source_path.parent) in result.stderr.replace("\n", "")


def test_the_stopped_check_is_quiet_when_stopping_ends_the_bill(config, no_timer):
    engine = Engine(config, provider=FakeProvider(config, CredentialResolver(config.project_dir)))
    engine.setup(wait=False)
    sweep_mod.ProjectRegistry().register(config.source_path)
    engine.stop()
    assert sweep_mod.stopped_and_billing() == []


def test_vast_maps_the_disk_bill_of_a_stopped_instance():
    from gpu4agents.providers.vast import mapper

    raw = {"id": 1, "actual_status": "exited", "cur_state": "stopped", "storage_cost": 0.2, "disk_space": 80.0}
    instance = mapper.map_instance(raw, {})
    assert instance.stopped_hourly_cost_usd == pytest.approx(16.0 / 730.0, rel=1e-3)
    assert type(instance).from_dict(instance.to_dict()).stopped_hourly_cost_usd == instance.stopped_hourly_cost_usd


def test_agent_install_asks_once_for_the_sweep_timer(no_timer, monkeypatch, tmp_path, jambu_home):
    from gpu4agents.agent import install as install_mod

    (jambu_home).mkdir(parents=True, exist_ok=True)
    (jambu_home / "credentials").write_text("VAST_API_KEY=from-file\n")
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    enabled = []
    monkeypatch.setattr(install_mod, "_enable_timer", lambda result: enabled.append(result) or True)
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["--json", "agent", "install", "-t", "claude", "--sweep-timer"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["sweep_timer"]["installed"] is True
    assert enabled and Path(enabled[0]["timer"]).is_file()


def test_agent_install_without_a_tty_or_flag_leaves_the_timer_and_says_so(no_timer, monkeypatch, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["--json", "agent", "install", "-t", "claude"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["sweep_timer"]["installed"] is False
    assert "gpu sweep --install-timer" in payload["sweep_timer"]["next"]
    assert not timer_mod.unit_paths()[1].exists()


# -- G2: the project lives in the repo, and an orphan can be taken over -------------


def test_ephemeral_dirs_are_recognised(tmp_path, monkeypatch):
    from gpu4agents.core.paths import ephemeral_reason

    monkeypatch.delenv("GPU_ALLOW_TEMP_PROJECT", raising=False)
    assert ephemeral_reason(Path("/tmp/claude-1000/x/scratchpad/heroimg"))
    assert ephemeral_reason(Path("/var/tmp/proj"))
    assert ephemeral_reason(Path("/home/u/work/scratchpad/p"))
    assert ephemeral_reason(Path("/home/u/repos/site")) is None


def test_init_refuses_a_temp_or_scratchpad_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("GPU_ALLOW_TEMP_PROJECT", raising=False)
    target = tmp_path / "scratchpad" / "heroimg"
    target.mkdir(parents=True)
    monkeypatch.chdir(target)
    result = runner.invoke(app, ["init", "--jobs"])
    assert result.exit_code == 2
    text = " ".join(result.output.split())
    assert "repository" in text and "--allow-temp" in text
    assert not (target / "gpu.yaml").exists()

    result = runner.invoke(app, ["init", "--jobs", "--allow-temp"])
    assert result.exit_code == 0, result.output
    assert (target / "gpu.yaml").exists()


def _orphan_from(config, tmp_path):
    """Take 1's situation: a machine set up from a project whose folder is gone."""
    provider = FakeProvider(config, CredentialResolver(config.project_dir))
    lost = write_project(tmp_path / "lost-session")
    old = Engine(lost, provider=provider)
    state = old.setup(wait=False)
    with old.store.transaction() as stored:
        stored.state = "idle"
    token, api_key = state.watchdog.token, state.api_key
    import shutil

    shutil.rmtree(lost.project_dir)
    return provider, state.instance_id, token, api_key


def test_adopt_takes_over_an_orphan_with_its_watchdog_and_key(config, tmp_path, monkeypatch):
    provider, instance_id, token, api_key = _orphan_from(config, tmp_path)
    provider.boot_scripts = {instance_id: provider.last_spec.onstart}
    engine = Engine(config, provider=provider)
    result = engine.adopt(instance_id)
    assert result["instance_id"] == instance_id
    assert result["reusable"] is True
    state = engine.store.load()
    assert state.instance_id == instance_id
    assert state.watchdog.token == token
    assert state.api_key == api_key
    # Same gpu.yaml as the lost project: setup reuses it instead of renting.
    again = engine.setup(wait=False)
    assert again.instance_id == instance_id and provider.counter == 1
    assert sweep_mod.ProjectRegistry().paths() == [config.source_path.resolve()]


def test_adopt_without_the_boot_script_can_still_stop_and_destroy(config, tmp_path):
    provider, instance_id, _, _ = _orphan_from(config, tmp_path)
    engine = Engine(config, provider=provider)
    result = engine.adopt(instance_id)
    assert result["reusable"] is False
    assert "gpu destroy" in result["next"]
    engine.destroy()
    assert instance_id not in provider.instances


def test_adopt_refuses_another_projects_machine_or_a_foreign_one(config, tmp_path):
    from gpu4agents.core.errors import ConfigError

    provider = FakeProvider(config, CredentialResolver(config.project_dir))
    other = write_project(tmp_path / "other")
    owned = Engine(other, provider=provider).setup(wait=False)
    sweep_mod.ProjectRegistry().register(other.source_path)
    engine = Engine(config, provider=provider)
    with pytest.raises(ConfigError, match="already tracks"):
        engine.adopt(owned.instance_id)
    with pytest.raises(ConfigError, match="no machine"):
        engine.adopt("i-404")


def test_adopt_refuses_when_this_project_already_has_a_machine(config, tmp_path):
    from gpu4agents.core.errors import StateError

    provider, instance_id, _, _ = _orphan_from(config, tmp_path)
    engine = Engine(config, provider=provider)
    engine.setup(wait=False)
    with pytest.raises(StateError):
        engine.adopt(instance_id)


def test_boot_script_recovery_reads_the_token_and_key_but_never_the_provider_key():
    from gpu4agents.watchdog.builder import build_onstart_script, recover_boot_secrets

    script = build_onstart_script(
        agent_source="print(1)",
        agent_config={"token": "tok-123", "port": 8777, "env": {"VAST_API_KEY": "provider-secret"}},
        runtime_start_command="true",
        runtime_env={"JAMBU_API_KEY": "model-key", "OTHER": "x"},
    )
    found = recover_boot_secrets(script, api_key_env="JAMBU_API_KEY")
    assert found == {"token": "tok-123", "port": 8777, "api_key": "model-key"}
    assert "provider-secret" not in json.dumps(found)
    assert recover_boot_secrets("#!/bin/bash\ntrue\n", api_key_env="JAMBU_API_KEY") == {}


def test_vast_hands_back_the_boot_script(config, monkeypatch):
    from gpu4agents.providers.vast.provider import VastProvider
    from gpu4agents.core.models import Instance

    provider = VastProvider(config, CredentialResolver(config.project_dir))
    class Stub:
        def get_instance(self, iid):
            return {"id": iid, "onstart": "#!/bin/bash"}

    provider._client = Stub()
    assert provider.boot_script(Instance(id="7", provider="vast")) == "#!/bin/bash"


# -- G4: the card the user approved is the card that is rented ------------------------


@pytest.fixture
def market(config):
    """Take 2's market: the cheapest card quoted, a dearer different card behind it."""
    provider = FakeProvider(config, CredentialResolver(config.project_dir))
    provider.hosts = ["m1", "m2", "m3"]
    provider.host_gpus = {"m1": "RTX 3060", "m2": "RTX 2060", "m3": "RTX 3060"}
    provider.host_prices = {"m1": 0.076, "m2": 0.094, "m3": 0.30}
    return provider


def test_offers_are_remembered_so_setup_can_pin_the_quoted_one(config, market):
    engine = Engine(config, provider=market)
    quoted = engine.offers(limit=5)
    assert quoted[0]["id"] == "offer-m1" and quoted[0]["gpu"] == "RTX 3060"
    state = engine.setup(wait=False, offer_id="offer-m1")
    assert state.instance.gpu_name == "RTX 3060"
    rented = engine.rented_vs_quoted(state)
    assert rented["matches_quote"] is True
    assert rented["quoted"] == {"offer_id": "offer-m1", "gpu": "RTX 3060", "hourly_cost_usd": 0.076}


def test_a_pinned_setup_never_falls_back_to_another_card(config, market):
    """The 3060 quoted is gone; the 2060 behind it must not be rented instead."""
    engine = Engine(config, provider=market)
    engine.offers(limit=5)
    market.hosts = ["m2", "m3"]  # m1 was taken between the quote and the yes
    from gpu4agents.core.errors import CapacityUnavailableError

    with pytest.raises(CapacityUnavailableError, match="approved"):
        engine.setup(wait=False, offer_id="offer-m1")
    assert market.rented_hosts == []


def test_a_pinned_setup_never_pays_more_than_the_quote(config, market):
    engine = Engine(config, provider=market)
    engine.offers(limit=5)
    market.hosts = ["m3"]  # only a dearer 3060 is left
    from gpu4agents.core.errors import CapacityUnavailableError

    with pytest.raises(CapacityUnavailableError):
        engine.setup(wait=False, offer_id="offer-m1")
    assert market.rented_hosts == []


def test_a_pinned_host_retry_stays_on_the_quoted_card(write_config, monkeypatch):
    config = write_config(lifecycle={"watchdog": {"enabled": False}})
    market = FakeProvider(config, CredentialResolver(config.project_dir))
    market.hosts = ["m1", "m2", "m3"]
    market.host_gpus = {"m1": "RTX 3060", "m2": "RTX 2060", "m3": "RTX 3060"}
    market.host_prices = {"m1": 0.076, "m2": 0.05, "m3": 0.076}
    monkeypatch.setattr("gpu4agents.core.engine.wait_for_healthy", lambda *a, **k: None)
    monkeypatch.setattr("gpu4agents.core.engine.time.sleep", lambda _s: None)
    engine = Engine(config, provider=market)
    engine.offers(limit=5)
    market.fail_first_hosts(1)  # m1, the quoted machine, fails to boot
    state = engine.setup(offer_id="offer-m1")
    assert state.instance.host_id == "m3" and state.instance.gpu_name == "RTX 3060"
    assert "m2" not in market.rented_hosts  # cheaper, but not the card approved


def test_an_unpinned_setup_says_when_it_rented_something_else(config, market):
    engine = Engine(config, provider=market)
    market.searches.clear()
    market.hosts = ["m1", "m2", "m3"]
    original = market.setup

    def take_m1_away(spec):
        market.hosts = ["m2", "m3"]  # gone between the budget check and the rent
        return original(spec)

    market.setup = take_m1_away
    state = engine.setup(wait=False)
    rented = engine.rented_vs_quoted(state)
    assert rented["matches_quote"] is False
    assert rented["rented"]["gpu"] == "RTX 2060"
    assert "RTX 3060" in rented["warning"] and "RTX 2060" in rented["warning"]


def test_setup_json_carries_rented_and_quoted(config, market, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    engine = Engine(config, provider=market)
    monkeypatch.setattr(_common.Context, "engine", lambda self: engine)
    result = runner.invoke(app, ["--json", "offers"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["--json", "setup", "--no-wait", "--offer", "offer-m1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["rented"]["gpu"] == "RTX 3060"
    assert payload["matches_quote"] is True


def test_setup_with_an_unknown_offer_asks_for_a_new_quote(config, market):
    from gpu4agents.core.errors import ConfigError

    engine = Engine(config, provider=market)
    with pytest.raises(ConfigError, match="gpu offers"):
        engine.setup(wait=False, offer_id="offer-nope")


# -- G5: a machine that vanished leaves a reason behind ------------------------------


def _vanished(write_config, idle_for_s, **lifecycle):
    """Take 1: ready, idle, nobody watching; then gone at the provider."""
    config = write_config(lifecycle={"idle_timeout": "15m", **lifecycle})
    engine = billing_engine(config)
    engine.setup(wait=False)
    with engine.store.transaction() as stored:
        stored.state = "idle"
        stored.starting = False
        stored.last_activity = utc_now() - idle_for_s
    engine.provider.instances.clear()
    engine.reconcile()
    return engine, [r for r in engine.events.tail(limit=50)]


def test_a_machine_gone_after_its_idle_timeout_is_recorded_as_the_watchdogs_doing(write_config):
    engine, events = _vanished(write_config, idle_for_s=40 * 60)
    [destroyed] = [r for r in events if r["event"] == "instance_destroyed"]
    assert destroyed["reason"] == "idle_timeout"
    assert destroyed["by"] == "watchdog" and destroyed["inferred"] is True
    [reconciled] = [r for r in events if r["event"] == "state_reconciled"]
    assert reconciled["likely_reason"] == "idle_timeout (watchdog)"


def test_a_machine_gone_while_in_use_is_not_blamed_on_the_watchdog(write_config):
    engine, events = _vanished(write_config, idle_for_s=60)
    assert not [r for r in events if r["event"] == "instance_destroyed"]
    [reconciled] = [r for r in events if r["event"] == "state_reconciled"]
    assert reconciled["likely_reason"].startswith("unknown")


def test_a_watchdog_that_only_stops_is_not_credited_with_a_destroy(write_config):
    engine, events = _vanished(write_config, idle_for_s=40 * 60, keep_disk=True)
    assert not [r for r in events if r["event"] == "instance_destroyed"]


def test_setup_brings_the_watchdogs_own_events_home(config, monkeypatch):
    """Take 1's log had no watchdog_identified: the watchdog's events only came
    home on a later command, and there was none before the machine went."""
    engine = billing_engine(config)

    class Watchdog:
        def events(self, limit=100):
            return [{"ts": "2026-11-11T00:29:40Z", "event": "watchdog_identified", "source": "watchdog"}]

        def __getattr__(self, name):
            return lambda *a, **k: True

    monkeypatch.setattr(engine, "beacon", lambda state: Watchdog())
    monkeypatch.setattr("gpu4agents.core.engine.wait_for_healthy", lambda *a, **k: None)
    monkeypatch.setattr("gpu4agents.core.engine.time.sleep", lambda _s: None)
    engine.setup()
    assert [r for r in engine.events.tail(limit=100) if r["event"] == "watchdog_identified"]


# -- G3: size for the peak, compare by the cost of the whole job ---------------------


def test_known_image_models_have_a_vram_floor():
    from gpu4agents.core.model_inspect import known_vram_floor

    assert known_vram_floor("stabilityai/stable-diffusion-xl-base-1.0") == (16, 24)
    assert known_vram_floor("black-forest-labs/FLUX.1-dev") == (24, 32)
    assert known_vram_floor("Qwen/Qwen2.5-7B-Instruct") is None


def test_validate_warns_when_the_card_is_sized_for_the_weights_not_the_peak(write_config):
    """Take 2: min_vram_gb 9 for SDXL; a 12 GB card ran out of memory decoding 1344x768."""
    config = write_config(
        model={"id": "stabilityai/stable-diffusion-xl-base-1.0", "task": "image"},
        runtime={"engine": "hf", "port": 8000},
        compute={"gpu": {"min_vram_gb": 9, "count": 1}, "disk_gb": 80},
    )
    engine = Engine(config, provider=FakeProvider(config, CredentialResolver(config.project_dir)))
    warnings = [i.message for i in engine.validate(check_provider=False).issues if i.field == "compute.gpu.min_vram_gb"]
    assert warnings and "16" in warnings[0] and "24" in warnings[0]


def test_init_sizes_a_known_image_model_for_its_peak(tmp_path, monkeypatch):
    import gpu4agents.cli.setup as setup_cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        setup_cli, "detect_serving", lambda model, given: ("hf", "image", 9, "")
    )
    result = runner.invoke(app, ["init", "--model", "stabilityai/stable-diffusion-xl-base-1.0"])
    assert result.exit_code == 0, result.output
    assert "min_vram_gb: 16" in (tmp_path / "gpu.yaml").read_text()


def test_offers_for_a_job_rank_by_the_cost_of_the_whole_job(config):
    """A card 4x faster at 3x the price costs less per job, and finishes sooner."""
    provider = FakeProvider(config, CredentialResolver(config.project_dir))
    provider.hosts = ["slow", "fast", "mystery"]
    provider.host_gpus = {"slow": "RTX 3060", "fast": "RTX 4090", "mystery": "Unknown X1"}
    provider.host_prices = {"slow": 0.10, "fast": 0.35, "mystery": 0.01}
    engine = Engine(config, provider=provider)
    rows = engine.offers(limit=5, for_minutes_on_4090=30)
    assert [r["gpu"] for r in rows] == ["RTX 4090", "RTX 3060", "Unknown X1"]
    fast, slow, mystery = rows
    assert fast["est_minutes"] == 30 and fast["est_job_cost_usd"] == pytest.approx(0.175)
    assert slow["est_minutes"] > 100 and slow["est_job_cost_usd"] > fast["est_job_cost_usd"]
    assert mystery["est_minutes"] is None and mystery["est_job_cost_usd"] is None
