from __future__ import annotations

import yaml
import pytest

from gpu4agents.core.config import CONFIG_FILENAMES, find_config, load_config
from gpu4agents.core.durations import format_duration, parse_duration
from gpu4agents.core.errors import ConfigError

from conftest import BASE_CONFIG


@pytest.mark.parametrize(
    "text,seconds",
    [("30s", 30), ("15m", 900), ("6h", 21600), ("1h30m", 5400), ("2d", 172800), ("90", 90)],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_parse_duration_none_means_unbounded():
    assert parse_duration(None) is None
    assert parse_duration("never") is None


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_format_duration():
    assert format_duration(5400) == "1h30m"
    assert format_duration(45) == "45s"
    assert format_duration(None) == "-"


def test_loads_the_spec_example(config):
    assert config.provider.name == "vast"
    assert config.model.id == "empero-ai/Qwythos-9B-v2"
    assert config.idle_timeout_s == 900
    assert config.max_lifetime_s == 21600
    assert config.runtime.port == 8000


def test_credentials_are_rejected_in_config(write_config):
    with pytest.raises(ConfigError) as exc:
        write_config(provider={"name": "vast", "api_key": "secret"})
    assert "api_key" in str(exc.value)


def test_idle_timeout_may_not_exceed_max_lifetime(write_config):
    with pytest.raises(ConfigError) as exc:
        write_config(lifecycle={"idle_timeout": "8h", "max_lifetime": "1h"})
    assert "idle_timeout" in str(exc.value)


def test_unknown_field_is_an_error(write_config):
    with pytest.raises(ConfigError):
        write_config(lifecycle={"idle_timout": "15m"})


def test_unsupported_version(write_config):
    with pytest.raises(ConfigError):
        write_config(version=2)


def test_missing_config_is_a_readable_error(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "nope.yml")
    assert "not found" in str(exc.value)


def test_env_interpolation(write_config, monkeypatch):
    monkeypatch.setenv("JAMBU_TEST_MODEL", "org/model-x")
    config = write_config(model={"id": "${JAMBU_TEST_MODEL}"})
    assert config.model.id == "org/model-x"


def test_config_is_found_from_a_subdirectory(config, tmp_path):
    nested = tmp_path / "experiments" / "deep"
    nested.mkdir(parents=True)
    assert find_config(nested) == config.source_path


def test_fingerprint_tracks_runtime_identity(write_config):
    base = write_config()
    same = write_config()
    assert base.fingerprint() == same.fingerprint()

    other_model = write_config(model={"id": "org/other"})
    assert other_model.fingerprint() != base.fingerprint()

    bigger_gpu = write_config(compute={"gpu": {"min_vram_gb": 48, "count": 1}})
    assert bigger_gpu.fingerprint() != base.fingerprint()

    # Lifecycle policy is not part of runtime identity: changing it must not
    # force a reprovision.
    looser = write_config(lifecycle={"idle_timeout": "30m"})
    assert looser.fingerprint() == base.fingerprint()


def test_state_dir_sits_next_to_the_config(config):
    assert config.state_dir == config.source_path.parent / ".gpu"


# -- jambu.yaml shared-manifest namespacing ----------------------------------


def test_legacy_flat_config_yml_still_works(write_flat_config):
    config = write_flat_config()
    assert config.model.id == "empero-ai/Qwythos-9B-v2"
    assert config.source_path.name == "config.yml"


def test_jambu_yaml_ignores_other_tools_top_level_keys(tmp_path):
    doc = {
        "some_other_jambu_tool": {"port": 4000, "unrelated": True},
        "gpu_runtime": BASE_CONFIG,
    }
    path = tmp_path / "jambu.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    config = load_config(path)
    assert config.model.id == "empero-ai/Qwythos-9B-v2"


def test_gpu_runtime_key_must_be_a_mapping(tmp_path):
    path = tmp_path / "jambu.yaml"
    path.write_text(yaml.safe_dump({"gpu_runtime": "not-a-mapping"}))
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "gpu_runtime" in str(exc.value)


def test_a_bare_document_in_jambu_yaml_is_treated_as_flat(tmp_path):
    """No gpu_runtime: key at all -> the whole document is the config, same
    rule as config.yml. Lets a solo project use the jambu.yaml filename
    without the wrapper if it never needs to share the file.
    """
    path = tmp_path / "jambu.yaml"
    path.write_text(yaml.safe_dump(BASE_CONFIG, sort_keys=False))
    config = load_config(path)
    assert config.model.id == "empero-ai/Qwythos-9B-v2"


def test_jambu_yaml_is_preferred_over_legacy_config_yml(tmp_path):
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({**BASE_CONFIG, "model": {"id": "org/legacy"}})
    )
    (tmp_path / "jambu.yaml").write_text(
        yaml.safe_dump({"gpu_runtime": {**BASE_CONFIG, "model": {"id": "org/current"}}})
    )
    config = load_config(start=tmp_path)
    assert config.model.id == "org/current"
    assert config.source_path.name == "jambu.yaml"


def test_config_filenames_search_order_favors_the_shared_manifest():
    assert CONFIG_FILENAMES.index("jambu.yaml") < CONFIG_FILENAMES.index("config.yml")


def test_unquoted_yaml_on_off_are_accepted_for_tool_calling(tmp_path):
    """YAML 1.1 reads bare on/off as booleans (the "Norway problem") - a user
    writing `tool_calling: on` unquoted must not get a confusing type error.
    """
    doc = {"gpu_runtime": {**BASE_CONFIG, "runtime": {"tool_calling": True}}}
    path = tmp_path / "jambu.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    assert load_config(path).runtime.tool_calling == "on"

    doc["gpu_runtime"]["runtime"]["tool_calling"] = False
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    assert load_config(path).runtime.tool_calling == "off"


# -- model catalog / profiles (sustainability: model-specific values live in
# one shared catalog, not duplicated across every project's full config) ----


def test_model_profile_merges_catalog_fragment_under_project_overrides(tmp_path):
    (tmp_path / "jambu.models.yaml").write_text(
        yaml.safe_dump(
            {
                "qwythos-9b": {
                    "model": {"id": "empero-ai/Qwythos-9B-v2", "trust_remote_code": True},
                    "compute": {"gpu": {"min_vram_gb": 40, "name_filter": "A6000"}},
                    "runtime": {"tool_calling": "on", "tool_call_parser": "qwen3_xml"},
                }
            }
        )
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "jambu.yaml").write_text(
        yaml.safe_dump(
            {
                "gpu_runtime": {
                    "version": 1,
                    "model_profile": "qwythos-9b",
                    "compute": {"gpu": {"min_vram_gb": 48}},  # overrides just this leaf
                    "lifecycle": {"idle_timeout": "15m", "max_lifetime": "6h"},
                }
            }
        )
    )

    config = load_config(project_dir / "jambu.yaml")
    assert config.model.id == "empero-ai/Qwythos-9B-v2"
    assert config.model.trust_remote_code is True
    assert config.compute.gpu.min_vram_gb == 48  # project override wins
    assert config.compute.gpu.name_filter == "A6000"  # from the profile, untouched
    assert config.runtime.tool_call_parser == "qwen3_xml"
    assert config.profile_name == "qwythos-9b"
    assert config.catalog_path == (tmp_path / "jambu.models.yaml").resolve()


def test_catalog_is_shared_by_sub_projects(tmp_path):
    """One catalog at a repo root serves every sub-project under it - the
    whole point: stop duplicating full configs into every directory.
    """
    (tmp_path / "jambu.models.yaml").write_text(
        yaml.safe_dump({"m": {"model": {"id": "org/model"}}})
    )
    for sub in ("a", "b"):
        d = tmp_path / sub
        d.mkdir()
        (d / "jambu.yaml").write_text(
            yaml.safe_dump({"gpu_runtime": {"version": 1, "model_profile": "m"}})
        )
    assert load_config(tmp_path / "a" / "jambu.yaml").model.id == "org/model"
    assert load_config(tmp_path / "b" / "jambu.yaml").model.id == "org/model"


def test_model_profile_without_a_catalog_is_a_clear_error(tmp_path):
    (tmp_path / "jambu.yaml").write_text(
        yaml.safe_dump({"gpu_runtime": {"version": 1, "model_profile": "nope"}})
    )
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "jambu.yaml")
    assert "jambu.models.yaml" in str(exc.value)


def test_unknown_profile_name_lists_whats_available(tmp_path):
    (tmp_path / "jambu.models.yaml").write_text(
        yaml.safe_dump({"a": {"model": {"id": "x/y"}}, "b": {"model": {"id": "x/z"}}})
    )
    (tmp_path / "jambu.yaml").write_text(
        yaml.safe_dump({"gpu_runtime": {"version": 1, "model_profile": "c"}})
    )
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "jambu.yaml")
    assert "a" in str(exc.value) and "b" in str(exc.value)


def test_profile_fragment_must_be_a_mapping(tmp_path):
    (tmp_path / "jambu.models.yaml").write_text(yaml.safe_dump({"broken": "not-a-mapping"}))
    (tmp_path / "jambu.yaml").write_text(
        yaml.safe_dump({"gpu_runtime": {"version": 1, "model_profile": "broken"}})
    )
    with pytest.raises(ConfigError):
        load_config(tmp_path / "jambu.yaml")


def test_no_model_profile_means_no_catalog_lookup_at_all(write_config):
    """A project that declares model:/compute: directly (no model_profile)
    must not require a catalog to exist - fully backward compatible.
    """
    config = write_config()
    assert config.profile_name is None
    assert config.catalog_path is None


def test_deep_merge_replaces_lists_wholesale_not_concatenated():
    from gpu4agents.core.config import deep_merge

    base = {"runtime": {"extra_args": ["--a"], "port": 8000}}
    override = {"runtime": {"extra_args": ["--b", "--c"]}}
    merged = deep_merge(base, override)
    assert merged["runtime"]["extra_args"] == ["--b", "--c"]
    assert merged["runtime"]["port"] == 8000


# -- fingerprint completeness (found live: changing runtime.tool_calling on
# an existing instance had NO effect because the onstart script - and the
# vllm command it launches - is baked in once at creation and re-executed
# verbatim on every restart; fingerprint is the ONLY thing that detects a
# runtime-affecting change and forces --recreate) -----------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime": {"tool_calling": "on", "tool_call_parser": "qwen3_xml"}},
        {"runtime": {"tool_call_parser": "hermes"}},
        {"runtime": {"dtype": "float16"}},
        {"runtime": {"gpu_memory_utilization": 0.8}},
        {"runtime": {"api_key": "secret-key"}},
        {"runtime": {"tool_server": "demo"}},
        {"runtime": {"vllm_args": {"max_num_seqs": 32}}},
        {"model": {"id": "empero-ai/Qwythos-9B-v2", "trust_remote_code": True}},
        {"model": {"id": "empero-ai/Qwythos-9B-v2", "served_name": "qwythos"}},
    ],
)
def test_every_server_args_affecting_field_changes_the_fingerprint(write_config, overrides):
    base = write_config()
    changed = write_config(**overrides)
    assert changed.fingerprint() != base.fingerprint(), (
        f"{overrides} changes the actual vllm command line but not the "
        "fingerprint - an existing instance would silently keep serving "
        "the old command forever"
    )


def test_lifecycle_and_budget_still_do_not_affect_the_fingerprint(write_config):
    """Runtime identity is about the served model, not lifecycle policy -
    changing idle_timeout must not force a reprovision.
    """
    base = write_config()
    changed = write_config(
        lifecycle={"idle_timeout": "45m", "max_lifetime": "12h"},
        budget={"max_hourly_cost_usd": 5.0},
    )
    assert changed.fingerprint() == base.fingerprint()


def test_stopped_ttl_defaults_to_a_day_and_null_disables_it(write_config):
    assert write_config().stopped_ttl_s == 24 * 3600
    assert write_config(lifecycle={"stopped_ttl": "2h"}).stopped_ttl_s == 7200
    assert write_config(lifecycle={"stopped_ttl": None}).stopped_ttl_s is None


def test_stopped_ttl_must_be_a_duration(write_config):
    with pytest.raises(ConfigError) as exc:
        write_config(lifecycle={"stopped_ttl": "tomorrow"})
    assert "stopped_ttl" in str(exc.value)


def test_traffic_counts_as_activity_by_default(write_config):
    assert write_config().lifecycle.traffic_counts_as_activity is True
    assert (
        write_config(lifecycle={"traffic_counts_as_activity": False})
        .lifecycle.traffic_counts_as_activity
        is False
    )


def test_init_templates_spell_out_the_new_lifecycle_defaults():
    import yaml

    from gpu4agents.core.config import DEFAULT_CONFIG_TEMPLATE, DEFAULT_CONFIG_TEMPLATE_FLAT

    nested = yaml.safe_load(DEFAULT_CONFIG_TEMPLATE.format(model_id="org/m"))["gpu_runtime"]
    flat = yaml.safe_load(DEFAULT_CONFIG_TEMPLATE_FLAT.format(model_id="org/m"))
    for doc in (nested, flat):
        assert doc["lifecycle"]["stopped_ttl"] == "24h"
        assert doc["lifecycle"]["traffic_counts_as_activity"] is True


def test_the_example_manifest_loads_with_the_new_fields():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "jambu.example.yaml"
    loaded = load_config(example)
    assert loaded.stopped_ttl_s == 24 * 3600
    assert loaded.lifecycle.traffic_counts_as_activity is True
    assert loaded.boot_timeout_s == 30 * 60
    assert loaded.compute.host_attempts == 3


def test_boot_and_startup_timeouts_are_separate(write_config):
    defaults = write_config()
    assert defaults.boot_timeout_s == 30 * 60
    assert defaults.startup_timeout_s == 10 * 60
    custom = write_config(health={"boot_timeout": "30m", "startup_timeout": "25m"})
    assert (custom.boot_timeout_s, custom.startup_timeout_s) == (1800.0, 1500.0)
    with pytest.raises(ConfigError, match="boot_timeout"):
        write_config(health={"boot_timeout": "soon"})


def test_host_attempts_must_allow_at_least_one_machine(write_config):
    assert write_config().compute.host_attempts == 3
    assert write_config(compute={"host_attempts": 1}).compute.host_attempts == 1
    with pytest.raises(ConfigError, match="host_attempts"):
        write_config(compute={"host_attempts": 0})


def test_host_attempts_and_boot_timeout_keep_the_instance_compatible(write_config):
    """Retry tuning is not part of the runtime: no `--recreate` for it."""
    base = write_config().fingerprint()
    assert write_config(compute={"host_attempts": 5}).fingerprint() == base
    assert write_config(health={"boot_timeout": "40m"}).fingerprint() == base
    assert write_config(health={"pull_stall_timeout": "5m"}).fingerprint() == base


def test_pull_stall_timeout_defaults_to_ten_minutes_and_can_be_off(write_config):
    assert write_config().pull_stall_timeout_s == 10 * 60
    assert write_config(health={"pull_stall_timeout": "off"}).pull_stall_timeout_s is None
    with pytest.raises(ConfigError, match="pull_stall_timeout"):
        write_config(health={"pull_stall_timeout": "soon"})
