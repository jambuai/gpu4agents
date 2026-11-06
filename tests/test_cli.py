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
    assert (tmp_path / "gpu.yaml").is_file()
    assert (tmp_path / ".env.example").is_file()

    validated = runner.invoke(app, ["validate", "--offline"])
    assert validated.exit_code == 0


def test_init_writes_gpu_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--model", "org/model"]).exit_code == 0
    assert (tmp_path / "gpu.yaml").is_file()
    assert not (tmp_path / "jambu.yaml").exists()


def test_init_stops_when_the_old_jambu_yaml_is_there(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "jambu.yaml").write_text("gpu_runtime: {}\n")
    result = runner.invoke(app, ["init", "--model", "org/model"])
    assert result.exit_code == 1
    assert "rename it to gpu.yaml" in result.output
    assert not (tmp_path / "gpu.yaml").exists()
    assert runner.invoke(app, ["init", "--model", "org/model", "--force"]).exit_code == 0
    assert (tmp_path / "gpu.yaml").is_file()
    assert (tmp_path / "jambu.yaml").read_text() == "gpu_runtime: {}\n"


def test_init_refuses_to_clobber(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    assert runner.invoke(app, ["init"]).exit_code == 1


def test_init_flat_writes_the_legacy_unnamespaced_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--flat", "--model", "org/model"])
    assert result.exit_code == 0
    assert (tmp_path / "config.yml").is_file()
    assert not (tmp_path / "gpu.yaml").exists()
    assert runner.invoke(app, ["validate", "--offline"]).exit_code == 0


def test_init_default_config_is_namespaced_under_gpu_runtime(tmp_path, monkeypatch):
    import yaml

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--model", "org/model"])
    doc = yaml.safe_load((tmp_path / "gpu.yaml").read_text())
    assert set(doc.keys()) == {"gpu_runtime"}
    assert doc["gpu_runtime"]["model"]["id"] == "org/model"


def test_providers_json_exposes_capabilities(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    result = runner.invoke(app, ["--json", "providers"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    by_name = {row["provider"]: row for row in payload["providers"]}
    assert {"vast", "runpod", "lambda"} <= set(by_name)
    vast = by_name["vast"]
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
    assert "no gpu.yaml" in result.output


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
    """No gpu.yaml anywhere - inspect-model must still work standalone."""
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
    assert (tmp_path / "gpu.models.yaml").is_file()

    import yaml

    catalog = yaml.safe_load((tmp_path / "gpu.models.yaml").read_text())
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
    (tmp_path / "gpu.models.yaml").write_text(
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


# -- gpu login ------------------------------------------------------------------


def _fake_account(monkeypatch, seen=None, fail=False):
    """The Vast account API, answering with a balance and the fields we must not print."""
    from gpu4agents.core.errors import AuthenticationError
    from gpu4agents.providers.vast import provider as vast_provider

    class Client:
        def __init__(self, key, **kwargs):
            self.key = key

        def whoami(self):
            if seen is not None:
                seen.append(self.key)
            if fail:
                raise AuthenticationError("vast.ai rejected the API key")
            return {"credit": 12.345, "email": "someone@example.com", "username": "someone"}

        def close(self):
            pass

    monkeypatch.setattr(vast_provider, "VastClient", Client)


def test_login_checks_the_key_then_saves_it_private(tmp_path, monkeypatch):
    import os
    import stat

    from gpu4agents.core.paths import gpu_home

    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    seen = []
    _fake_account(monkeypatch, seen)
    creds = gpu_home() / "credentials"
    creds.parent.mkdir(parents=True, exist_ok=True)
    creds.write_text("HF_TOKEN=hf_x\nVAST_API_KEY=old\n")

    result = runner.invoke(app, ["--json", "login", "--stdin"], input="new-key\n")

    assert result.exit_code == 0, result.output
    assert seen == ["new-key"]
    payload = json.loads(result.stdout)
    assert payload == {"ok": True, "provider": "vast", "saved": str(creds), "balance_usd": 12.35}
    assert "new-key" not in result.output and "someone" not in result.output
    assert creds.read_text() == "HF_TOKEN=hf_x\nVAST_API_KEY=new-key\n"
    assert stat.S_IMODE(os.stat(creds).st_mode) == 0o600


def test_login_rejects_a_bad_key_without_writing_it(tmp_path, monkeypatch):
    from gpu4agents.core.paths import gpu_home

    monkeypatch.chdir(tmp_path)
    _fake_account(monkeypatch, fail=True)
    result = runner.invoke(app, ["login", "--stdin"], input="bad\n")
    assert result.exit_code == 5
    assert not (gpu_home() / "credentials").exists()


def test_login_without_a_terminal_or_stdin_says_how(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_account(monkeypatch)
    result = runner.invoke(app, ["login"], input="")
    assert result.exit_code == 2
    assert "own terminal" in " ".join(result.output.split())


def test_login_check_reports_where_the_key_is(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    _fake_account(monkeypatch)
    missing = runner.invoke(app, ["login", "--check"])
    assert missing.exit_code == 5
    assert "gpu login" in missing.output

    monkeypatch.setenv("VAST_API_KEY", "from-env")
    found = runner.invoke(app, ["--json", "login", "--check"])
    assert found.exit_code == 0, found.output
    assert json.loads(found.stdout) == {
        "ok": True,
        "provider": "vast",
        "source": "environment",
        "balance_usd": 12.35,
    }


# -- other providers: plugins ---------------------------------------------------

ACME_PLUGIN = """
from tests.fakes import FakeProvider


class AcmeProvider(FakeProvider):
    name = "acme"
    display_name = "Acme Cloud"
    required_credentials = ["ACME_API_KEY"]
    credentials_url = "https://acme.example/keys"

    @classmethod
    def verify_credentials(cls, values):
        if values["ACME_API_KEY"] != "good":
            from gpu4agents.core.errors import AuthenticationError

            raise AuthenticationError("acme rejected the key")
        return {"balance_usd": 3.0}
"""


def _acme_project(tmp_path, monkeypatch):
    from gpu4agents.providers.registry import registry

    monkeypatch.setattr(registry, "_providers", dict(registry._providers))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    monkeypatch.delenv("GPU_PROVIDER", raising=False)
    assert runner.invoke(app, ["init", "--provider", "acme", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]).exit_code == 0
    (tmp_path / "gpu_providers").mkdir()
    (tmp_path / "gpu_providers" / "acme.py").write_text(ACME_PLUGIN)


def test_init_with_an_unknown_provider_points_jambu_yaml_at_its_adapter_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--provider", "acme"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "gpu.yaml").read_text()
    assert "    name: acme\n    plugin: ./gpu_providers/acme.py:AcmeProvider\n" in text
    assert "gpu providers new acme" in " ".join(result.output.split())
    assert "VAST_API_KEY" not in (tmp_path / ".env.example").read_text()


def test_providers_new_writes_an_adapter_that_loads(tmp_path, monkeypatch):
    from gpu4agents.providers.base import GPUProvider
    from gpu4agents.providers.registry import load_plugin

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["--json", "providers", "new", "my-cloud", "--display-name", "My Cloud"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["provider"] == {"name": "my-cloud", "plugin": "./gpu_providers/my-cloud.py:MyCloudProvider"}
    cls = load_plugin(payload["provider"]["plugin"], tmp_path)
    assert issubclass(cls, GPUProvider)
    assert (cls.name, cls.display_name, cls.required_credentials) == ("my-cloud", "My Cloud", ["MY_CLOUD_API_KEY"])
    import sys

    actions = sys.modules[cls.__module__].WATCHDOG_ACTIONS
    compile(actions, "watchdog_actions", "exec")
    assert "def provider_stop(ctx)" in actions and "def provider_destroy(ctx)" in actions
    # an unfinished adapter fails with a normal error, not a traceback
    from gpu4agents.providers.registry import registry

    monkeypatch.setattr(registry, "_providers", dict(registry._providers))
    assert runner.invoke(app, ["init", "--provider", "my-cloud"]).exit_code == 0
    unfinished = runner.invoke(app, ["login", "--stdin"], input="key\n")
    assert unfinished.exit_code == 9, unfinished.output
    assert "verify_credentials" in unfinished.output and "Traceback" not in unfinished.output
    again = runner.invoke(app, ["providers", "new", "my-cloud"])
    assert again.exit_code == 2  # refuses to overwrite
    assert runner.invoke(app, ["providers", "new", "vast"]).exit_code == 2


def test_a_plugin_provider_drives_the_cli_like_a_built_in_one(tmp_path, monkeypatch):
    _acme_project(tmp_path, monkeypatch)
    listed = json.loads(runner.invoke(app, ["--json", "providers"]).stdout)
    acme = next(row for row in listed["providers"] if row["provider"] == "acme")
    assert acme["selected"] and acme["missing_credentials"] == ["ACME_API_KEY"]

    missing = runner.invoke(app, ["login", "--check"])
    assert missing.exit_code == 5
    assert "ACME_API_KEY" in missing.output

    bad = runner.invoke(app, ["login", "--stdin"], input="nope\n")
    assert bad.exit_code == 5
    good = runner.invoke(app, ["--json", "login", "--stdin"], input="good\n")
    assert good.exit_code == 0, good.output
    assert json.loads(good.stdout)["provider"] == "acme"
    found = runner.invoke(app, ["--json", "login", "--check"])
    assert json.loads(found.stdout)["balance_usd"] == 3.0

    from gpu4agents.core.config import load_config
    from gpu4agents.core.engine import Engine

    engine = Engine(load_config(tmp_path / "gpu.yaml"))
    assert engine.provider.name == "acme"


def test_a_plugin_whose_name_differs_from_the_config_is_refused(tmp_path, monkeypatch):
    _acme_project(tmp_path, monkeypatch)
    text = (tmp_path / "gpu.yaml").read_text().replace("name: acme", "name: other")
    (tmp_path / "gpu.yaml").write_text(text)
    result = runner.invoke(app, ["login", "--check"])
    assert result.exit_code == 2
    assert "make them the same" in " ".join(result.output.split())


def test_login_with_provider_makes_it_the_default_outside_projects(tmp_path, monkeypatch):
    from gpu4agents.core.paths import gpu_home
    from gpu4agents.providers.registry import registry
    from tests.fakes import FakeProvider

    class Acme(FakeProvider):
        name = "acme2"
        display_name = "Acme"
        required_credentials = ["ACME2_KEY"]

    monkeypatch.setitem(registry._providers, "acme2", Acme)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GPU_PROVIDER", raising=False)
    result = runner.invoke(app, ["login", "--provider", "acme2", "--stdin"], input="k\n")
    assert result.exit_code == 0, result.output
    text = (gpu_home() / "credentials").read_text()
    assert "ACME2_KEY=k\n" in text and "GPU_PROVIDER=acme2\n" in text
    from gpu4agents.cli.account import default_provider_name

    assert default_provider_name() == "acme2"


def test_entry_point_providers_are_registered(monkeypatch):
    import importlib

    registry_mod = importlib.import_module("gpu4agents.providers.registry")
    from tests.fakes import FakeProvider

    class Installed(FakeProvider):
        name = "installed"

    class Entry:
        def load(self):
            return Installed

    monkeypatch.setattr(registry_mod.registry, "_providers", dict(registry_mod.registry._providers))
    monkeypatch.setattr(registry_mod.metadata, "entry_points", lambda group=None: [Entry()])
    assert "installed" in registry_mod.load_builtin_providers().names()


def test_login_first_time_shows_the_providers_and_the_steps(tmp_path, monkeypatch, capsys):
    from gpu4agents.cli import account
    from gpu4agents.providers.vast.provider import VastProvider

    monkeypatch.setattr(account.typer, "prompt", lambda *a, **k: "1")
    chosen = account._choose_provider(account.registry_classes(), VastProvider)
    assert chosen is VastProvider
    account._walk_through(chosen)
    out = " ".join(capsys.readouterr().out.split())
    assert "1) Vast.ai" in out and "Another provider I already use" in out
    assert "Create a Vast.ai account: https://cloud.vast.ai/" in out
    assert "billing" in out and "https://cloud.vast.ai/manage-keys/" in out


def test_login_another_provider_points_to_the_agent_and_saves_nothing(tmp_path, monkeypatch, capsys):
    from gpu4agents.cli import account
    from gpu4agents.providers.vast.provider import VastProvider

    classes = account.registry_classes()
    monkeypatch.setattr(account.typer, "prompt", lambda *a, **k: str(len(classes) + 1))
    assert account._choose_provider(classes, VastProvider) is None
    out = " ".join(capsys.readouterr().out.split())
    assert "providers.md" in out and "gpu login --provider NAME" in out


def test_gpu4agents_is_installed_as_a_second_command_name():
    from pathlib import Path

    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    scripts = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    assert 'gpu = "gpu4agents.cli.main:main"' in scripts
    assert 'gpu4agents = "gpu4agents.cli.main:main"' in scripts


def test_login_lists_every_built_in_provider(monkeypatch, capsys):
    from gpu4agents.cli import account
    from gpu4agents.providers.vast.provider import VastProvider

    monkeypatch.setattr(account.typer, "prompt", lambda *a, **k: "3")
    chosen = account._choose_provider(account.registry_classes(), VastProvider)
    out = " ".join(capsys.readouterr().out.split())
    assert "1) Vast.ai" in out and "2) Lambda" in out and "3) RunPod" in out
    assert "4) Another provider I already use" in out
    assert chosen.name == "runpod"
    account._walk_through(chosen)
    steps = " ".join(capsys.readouterr().out.split())
    assert "https://console.runpod.io/signup" in steps and "https://console.runpod.io/user/credentials" in steps


def test_init_uses_the_provider_login_connected(tmp_path, monkeypatch):
    from gpu4agents.core.credentials import save_user_credential

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GPU_PROVIDER", raising=False)
    save_user_credential("GPU_PROVIDER", "lambda")
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "gpu.yaml").read_text()
    assert "name: lambda\n" in text and "plugin:" not in text
    assert "LAMBDA_API_KEY=" in (tmp_path / ".env.example").read_text()
    assert "gpu login --provider lambda" in result.output


def test_runpod_and_lambda_are_built_in_names():
    from gpu4agents.core.errors import ConfigError
    from gpu4agents.providers import scaffold

    for name in ("runpod", "lambda"):
        try:
            scaffold.write_template(name, __import__("pathlib").Path("/nonexistent"))
        except ConfigError as exc:
            assert "built in" in str(exc)
        else:
            raise AssertionError(f"{name} must be refused")
