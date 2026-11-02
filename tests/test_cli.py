from __future__ import annotations

import json

from typer.testing import CliRunner

import gpu4agents
from gpu4agents.cli.main import app

runner = CliRunner()


def test_help_lists_the_spec_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("providers", "validate", "setup", "run", "status", "lock", "stop", "destroy"):
        assert command in result.stdout


def test_version_flag_and_command_print_the_package_version():
    for args in (["--version"], ["version"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert result.stdout.strip() == gpu4agents.__version__


def test_init_writes_a_valid_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--model", "org/model"])
    assert result.exit_code == 0
    assert (tmp_path / "jambu.yaml").is_file()
    assert (tmp_path / ".env.example").is_file()

    validated = runner.invoke(app, ["validate", "--offline"])
    assert validated.exit_code == 0


def test_init_refuses_to_clobber(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    assert runner.invoke(app, ["init"]).exit_code == 1


def test_init_flat_writes_the_legacy_unnamespaced_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--flat", "--model", "org/model"])
    assert result.exit_code == 0
    assert (tmp_path / "config.yml").is_file()
    assert not (tmp_path / "jambu.yaml").exists()
    assert runner.invoke(app, ["validate", "--offline"]).exit_code == 0


def test_init_default_config_is_namespaced_under_gpu_runtime(tmp_path, monkeypatch):
    import yaml

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--model", "org/model"])
    doc = yaml.safe_load((tmp_path / "jambu.yaml").read_text())
    assert set(doc.keys()) == {"gpu_runtime"}
    assert doc["gpu_runtime"]["model"]["id"] == "org/model"


def test_providers_json_exposes_capabilities(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    result = runner.invoke(app, ["--json", "providers"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    vast = payload["providers"][0]
    assert vast["provider"] == "vast"
    assert vast["status"] == "unavailable"
    assert vast["capabilities"]["stop"] is True
    assert vast["required_credentials"] == ["VAST_API_KEY"]


def test_validate_fails_without_credentials(config, monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.chdir(config.project_dir)
    result = runner.invoke(app, ["--json", "validate"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("VAST_API_KEY" in error["message"] for error in payload["errors"])


def test_missing_config_is_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 2
    assert "no jambu.yaml" in result.output


def test_status_json_on_a_fresh_project(config, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    result = runner.invoke(app, ["--json", "status"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state"] == "none"
    assert payload["provider"] == "vast"
    assert payload["capabilities"]["stop"] is True


def test_stop_without_an_instance_is_a_success(config, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    result = runner.invoke(app, ["--json", "stop"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["changed"] is False


def test_run_requires_a_command(config, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    assert runner.invoke(app, ["run"]).exit_code != 0


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == gpu4agents.__version__


def test_inspect_model_json(monkeypatch, tmp_path):
    import httpx
    import respx
    from gpu4agents.core.model_inspect import HF_API_BASE, HF_HUB_BASE

    monkeypatch.chdir(tmp_path)
    with respx.mock:
        respx.get(f"{HF_API_BASE}/Qwen/Qwen2.5-7B-Instruct").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "Qwen/Qwen2.5-7B-Instruct",
                    "gated": False,
                    "safetensors": {"parameters": {"BF16": 7615616512}, "total": 7615616512},
                },
            )
        )
        respx.get(f"{HF_HUB_BASE}/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json").mock(
            return_value=httpx.Response(
                200,
                json={
                    "architectures": ["Qwen2ForCausalLM"],
                    "model_type": "qwen2",
                    "torch_dtype": "bfloat16",
                },
            )
        )
        result = runner.invoke(app, ["--json", "inspect-model", "Qwen/Qwen2.5-7B-Instruct"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["suggested_tool_call_parser"] == "hermes"
    assert payload["suggested_min_vram_gb"] == 22


def test_inspect_model_does_not_require_a_project_config(monkeypatch, tmp_path):
    """No jambu.yaml anywhere - inspect-model must still work standalone."""
    import httpx
    import respx
    from gpu4agents.core.model_inspect import HF_API_BASE, HF_HUB_BASE

    monkeypatch.chdir(tmp_path)
    with respx.mock:
        respx.get(f"{HF_API_BASE}/org/model").mock(
            return_value=httpx.Response(200, json={"id": "org/model", "gated": False})
        )
        respx.get(f"{HF_HUB_BASE}/org/model/raw/main/config.json").mock(
            return_value=httpx.Response(404)
        )
        result = runner.invoke(app, ["inspect-model", "org/model"])
    assert result.exit_code == 0


def test_inspect_model_add_profile_writes_the_catalog(monkeypatch, tmp_path):
    import httpx
    import respx
    from gpu4agents.core.model_inspect import HF_API_BASE, HF_HUB_BASE

    monkeypatch.chdir(tmp_path)
    with respx.mock:
        respx.get(f"{HF_API_BASE}/Qwen/Qwen2.5-7B-Instruct").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "Qwen/Qwen2.5-7B-Instruct",
                    "gated": False,
                    "safetensors": {"parameters": {"BF16": 7615616512}, "total": 7615616512},
                },
            )
        )
        respx.get(f"{HF_HUB_BASE}/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json").mock(
            return_value=httpx.Response(
                200, json={"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}
            )
        )
        result = runner.invoke(
            app,
            ["inspect-model", "Qwen/Qwen2.5-7B-Instruct", "--add-profile", "qwen25-7b"],
        )
    assert result.exit_code == 0
    assert (tmp_path / "jambu.models.yaml").is_file()

    import yaml

    catalog = yaml.safe_load((tmp_path / "jambu.models.yaml").read_text())
    assert catalog["qwen25-7b"]["model"]["id"] == "Qwen/Qwen2.5-7B-Instruct"

    # Re-running without --force must not clobber it.
    with respx.mock:
        respx.get(f"{HF_API_BASE}/Qwen/Qwen2.5-7B-Instruct").mock(
            return_value=httpx.Response(200, json={"id": "x", "gated": False})
        )
        respx.get(f"{HF_HUB_BASE}/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json").mock(
            return_value=httpx.Response(404)
        )
        again = runner.invoke(
            app,
            ["inspect-model", "Qwen/Qwen2.5-7B-Instruct", "--add-profile", "qwen25-7b"],
        )
    assert again.exit_code != 0


def test_profiles_command_lists_the_catalog(monkeypatch, tmp_path):
    import yaml

    monkeypatch.chdir(tmp_path)
    (tmp_path / "jambu.models.yaml").write_text(
        yaml.safe_dump(
            {
                "qwythos-9b": {
                    "model": {"id": "empero-ai/Qwythos-9B-v2"},
                    "compute": {"gpu": {"min_vram_gb": 40}},
                    "runtime": {"tool_call_parser": "qwen3_xml"},
                }
            }
        )
    )
    result = runner.invoke(app, ["--json", "profiles"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "qwythos-9b" in payload["profiles"]


def test_profiles_command_with_no_catalog_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["profiles"])
    assert result.exit_code == 0


def test_logs_takes_a_source_and_a_line_count(config, monkeypatch):
    from gpu4agents.core.engine import Engine

    monkeypatch.chdir(config.project_dir)
    seen = {}

    def _read_logs(self, limit=200, source=None):
        seen.update(limit=limit, source=source)
        return (source or "runtime"), ["RuntimeError: driver too old"]

    monkeypatch.setattr(Engine, "read_logs", _read_logs)
    result = runner.invoke(app, ["--json", "logs", "--source", "onstart", "--lines", "7"])
    assert result.exit_code == 0, result.output
    assert seen == {"limit": 7, "source": "onstart"}
    assert json.loads(result.stdout) == {
        "source": "onstart",
        "lines": ["RuntimeError: driver too old"],
    }

    result = runner.invoke(app, ["logs"])
    assert result.exit_code == 0
    assert seen == {"limit": 200, "source": None}
    assert "driver too old" in result.stdout


def _saved_endpoint(config, api_key="k-123"):
    from gpu4agents.core.state import RuntimeState, StateStore

    store = StateStore(config.state_dir)
    state = RuntimeState(
        provider="vast",
        instance_id="42",
        state="ready",
        endpoint="http://1.2.3.4:5",
        api_key=api_key,
    )
    store.save(state)


def test_env_prints_what_an_openai_client_needs(config, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    monkeypatch.setenv("VAST_API_KEY", "x")
    _saved_endpoint(config)
    result = runner.invoke(app, ["env"])
    assert result.exit_code == 0
    assert "export OPENAI_BASE_URL=http://1.2.3.4:5/v1" in result.stdout
    assert "export OPENAI_API_KEY=k-123" in result.stdout

    payload = json.loads(runner.invoke(app, ["--json", "env"]).stdout)
    assert payload["base_url"] == "http://1.2.3.4:5/v1"
    assert payload["api_key"] == "k-123"
    assert payload["model"] == config.model.id


def test_env_without_an_endpoint_exits_3(config, monkeypatch):
    monkeypatch.chdir(config.project_dir)
    monkeypatch.setenv("VAST_API_KEY", "x")
    assert runner.invoke(app, ["env"]).exit_code == 3
    result = runner.invoke(app, ["env", "--format", "json"])
    assert result.exit_code == 3
    assert json.loads(result.stdout)["state"] == "none"
