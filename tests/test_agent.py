from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from gpu4agents.agent import install as install_mod
from gpu4agents.agent.jobs import Job
from gpu4agents.agent.mcp import PROTOCOL_VERSIONS, Server
from gpu4agents.cli.main import app


def _call(server: Server, name: str, **arguments):
    result = server.call_tool(name, arguments)
    return result["isError"], json.loads(result["content"][0]["text"])


def _rpc(server: Server, *messages) -> list[dict]:
    out = io.StringIO()
    server.serve(stdin=io.StringIO("".join(json.dumps(m) + "\n" for m in messages)), stdout=out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


# -- protocol ---------------------------------------------------------------


def test_initialize_negotiates_the_protocol_version(tmp_path):
    server = Server(project=tmp_path)
    for asked, expected in (("2025-03-26", "2025-03-26"), ("1999-01-01", PROTOCOL_VERSIONS[0])):
        reply = server.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": asked}}
        )
        assert reply["result"]["protocolVersion"] == expected
        assert reply["result"]["capabilities"] == {"tools": {"listChanged": False}}
        assert "instructions" in reply["result"]


def test_serve_speaks_newline_delimited_json_rpc(tmp_path):
    replies = _rpc(
        Server(project=tmp_path),
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        {"jsonrpc": "2.0", "id": 3, "method": "resources/list"},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope"}},
    )
    assert [r["id"] for r in replies] == [1, 2, 3, 4]  # the notification gets no reply
    assert replies[1]["result"] == {}
    assert replies[2]["error"]["code"] == -32601
    assert replies[3]["error"]["code"] == -32602


def test_a_bad_line_is_a_parse_error_not_a_crash(tmp_path):
    out = io.StringIO()
    stdin = io.StringIO('not json\n{"jsonrpc":"2.0","id":7,"method":"ping"}\n')
    Server(project=tmp_path).serve(stdin=stdin, stdout=out)
    first, second = [json.loads(line) for line in out.getvalue().splitlines()]
    assert first["error"]["code"] == -32700
    assert second["id"] == 7


def test_stray_prints_never_reach_the_protocol_stream(tmp_path, monkeypatch):
    server = Server(project=tmp_path)
    monkeypatch.setattr(server.tools["gpu_ls"], "handler", lambda args: print("noise") or {"ok": True})
    replies = _rpc(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "gpu_ls"}})
    assert replies[0]["result"]["isError"] is False


def test_every_tool_has_a_valid_schema_and_annotations(tmp_path):
    tools = Server(project=tmp_path).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"gpu_setup", "gpu_run", "gpu_stop", "gpu_destroy", "gpu_ls", "gpu_connection"} <= names
    for tool in tools:
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert "project_dir" in schema["properties"]
        assert set(schema.get("required", [])) <= set(schema["properties"])
        assert re.fullmatch(r"gpu_[a-z]+(_[a-z]+)?", tool["name"])
    by_name = {t["name"]: t for t in tools}
    assert by_name["gpu_destroy"]["annotations"]["destructiveHint"] is True
    assert by_name["gpu_status"]["annotations"]["readOnlyHint"] is True
    assert "COSTS MONEY" in by_name["gpu_setup"]["description"]


# -- tools against the real CLI (no provider calls) -------------------------


def test_init_then_status_then_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    server = Server(project=tmp_path)
    error, init = _call(server, "gpu_init", model="Qwen/Qwen2.5-0.5B-Instruct", max_hourly_cost_usd=0.4, min_vram_gb=16)
    assert not error, init
    doc = yaml.safe_load((tmp_path / "gpu.yaml").read_text())["gpu_runtime"]
    assert doc["model"]["id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert doc["budget"]["max_hourly_cost_usd"] == 0.4
    assert doc["compute"]["gpu"]["min_vram_gb"] == 16

    error, status = _call(server, "gpu_status")
    assert not error and status["state"] == "none"

    error, conn = _call(server, "gpu_connection")
    assert error and conn["exit_code"] == 3 and conn["next"] == "call gpu_setup"


def test_a_subdirectory_finds_the_project_upwards(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    sub = tmp_path / "src" / "deep"
    sub.mkdir(parents=True)
    error, status = _call(Server(project=sub), "gpu_status")
    assert not error and status["state"] == "none"


def test_no_project_says_how_to_make_one(tmp_path):
    error, payload = _call(Server(project=tmp_path), "gpu_status")
    assert error and payload["exit_code"] == 2 and "gpu_init" in payload["next"]


def test_a_job_needs_a_started_machine_and_never_rents_one(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    error, payload = _call(server, "gpu_job_run", command="python train.py")
    assert error and payload["exit_code"] == 3
    assert "gpu_setup" in payload["next"]

    error, listing = _call(server, "gpu_job_status")
    assert not error and listing == {"jobs": []}


def test_destroy_requires_an_explicit_confirm(tmp_path):
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    error, payload = _call(server, "gpu_destroy", confirm=False)
    assert error and "confirm=true" in payload["error"]


def test_destroy_with_confirm_on_an_empty_project_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    error, payload = _call(server, "gpu_destroy", confirm=True)
    assert not error, payload


# -- background setup ---------------------------------------------------------


def _fake_cli(server: Server, monkeypatch, script: str) -> None:
    """Make every gpu invocation run ``script`` instead (setup is the only user here)."""
    monkeypatch.setattr(server, "_argv", lambda args, json_output=True: [sys.executable, "-c", script])


def test_setup_runs_in_the_background_and_reattaches(tmp_path, monkeypatch):
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    marker = tmp_path / "starts"
    _fake_cli(
        server,
        monkeypatch,
        "import json,time,pathlib;"
        f"p=pathlib.Path({str(marker)!r});p.write_text(p.read_text()+'x' if p.exists() else 'x');"
        "time.sleep(1.5);"
        "print(json.dumps({'state':'ready','base_url':'http://h:1/v1','api_key':'k','model':'org/m','watchdog':{}}))",
    )
    error, first = _call(server, "gpu_setup", wait_seconds=0)
    assert not error and first["status"] == "running" and first["started_now"] is True

    error, again = _call(server, "gpu_setup", wait_seconds=0)
    assert again["status"] == "running" and again["started_now"] is False

    error, done = _call(server, "gpu_setup", wait_seconds=20)
    assert not error, done
    assert done["status"] == "ready" and done["base_url"] == "http://h:1/v1"
    assert "watchdog" not in done
    assert marker.read_text() == "x"  # one setup process, not three


def test_a_failed_setup_carries_the_exit_code_and_the_error(tmp_path, monkeypatch):
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    _fake_cli(server, monkeypatch, "import sys;sys.stderr.write('error: no offers match\\n');sys.exit(6)")
    error, payload = _call(server, "gpu_setup", wait_seconds=20)
    assert error
    assert payload["exit_code"] == 6
    assert payload["error"] == "no offers match"
    assert "capacity" in payload["meaning"]


def test_setup_status_shows_up_in_gpu_status(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    job = Job(tmp_path)
    job.start([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path)
    try:
        error, status = _call(server, "gpu_status")
        assert not error and status["setup_job"]["status"] == "running"
    finally:
        import os
        import signal

        os.killpg(int(job.meta()["pid"]), signal.SIGKILL)


def test_a_job_that_died_without_a_code_is_lost(tmp_path):
    job = Job(tmp_path)
    job.dir.mkdir(parents=True)
    job.meta_path.write_text(json.dumps({"pid": 2**22 + 12345, "argv": [], "started_at": 1.0}))
    assert job.report()["status"] == "lost"


# -- gpu_run ------------------------------------------------------------------


def test_run_returns_output_and_exit_code(tmp_path, monkeypatch):
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    _fake_cli(server, monkeypatch, "import sys;print('42 tokens');sys.exit(3)")
    error, result = _call(server, "gpu_run", command=["python", "eval.py"])
    assert not error
    assert result["exit_code"] == 3 and "42 tokens" in result["stdout"] and not result["timed_out"]


@pytest.mark.skipif(sys.platform == "win32", reason="process groups")
def test_run_interrupts_a_command_past_its_timeout(tmp_path, monkeypatch):
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    _fake_cli(server, monkeypatch, "import time;print('started',flush=True);time.sleep(60)")
    error, result = _call(server, "gpu_run", command=["x"], timeout_seconds=1)
    assert result["timed_out"] is True and "started" in result["stdout"]


# -- install ------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_install_for_claude_and_cursor_merges_existing_configs(tmp_path, home):
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    result = install_mod.install(["claude", "cursor"], project)

    skill = project / ".claude" / "skills" / "gpu4agents" / "SKILL.md"
    assert skill.read_text() == install_mod.skill_text()
    shared = project / ".agents" / "skills" / "gpu4agents" / "SKILL.md"
    assert shared.read_text() == install_mod.skill_text()
    servers = json.loads((project / ".mcp.json").read_text())["mcpServers"]
    assert servers["other"] == {"command": "x"}
    assert servers["gpu"]["args"][-1] == "mcp"
    cursor = json.loads((project / ".cursor" / "mcp.json").read_text())["mcpServers"]["gpu"]
    assert cursor["args"][-3:] == ["mcp", "--project", "${workspaceFolder}"]
    assert {i["action"] for i in result["installed"]} == {"created"}

    again = install_mod.install(["claude", "cursor"], project)
    assert {i["action"] for i in again["installed"]} == {"unchanged"}


def test_install_for_desktop_pins_the_project_and_zips_the_skill(tmp_path, home, monkeypatch):
    import zipfile

    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")
    project = tmp_path / "proj"
    project.mkdir()
    result = install_mod.install(["desktop"], project)
    config = json.loads((home / ".config" / "Claude" / "claude_desktop_config.json").read_text())
    entry = config["mcpServers"]["gpu"]
    assert entry["args"][-2:] == ["--project", str(project.resolve())]
    assert Path(entry["command"]).is_absolute()
    archive = Path(next(i["path"] for i in result["installed"] if "zip" in i["what"]))
    assert zipfile.ZipFile(archive).namelist() == ["gpu4agents/SKILL.md"]


def test_global_claude_install_uses_claude_mcp_add(tmp_path, home, monkeypatch):
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(install_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    install_mod.install(["claude"], tmp_path, global_scope=True, run=lambda argv, **kw: calls.append(argv) or Done())
    assert calls[-1][:6] == ["claude", "mcp", "add", "--scope", "user", "gpu"]
    assert calls[-1][-1] == "mcp"
    assert (home / ".claude" / "skills" / "gpu4agents" / "SKILL.md").is_file()


def test_install_refuses_a_broken_config_instead_of_overwriting_it(tmp_path, home):
    (tmp_path / ".mcp.json").write_text("{not json")
    from gpu4agents.core.errors import ConfigError

    with pytest.raises(ConfigError):
        install_mod.install(["claude"], tmp_path)
    assert (tmp_path / ".mcp.json").read_text() == "{not json"


def test_cli_agent_install_and_skill(tmp_path, home, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["--json", "agent", "install", "-t", "claude", "-t", "cursor"])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["installed"]) == 4
    assert runner.invoke(app, ["agent", "install", "-t", "vscode"]).exit_code == 2
    assert runner.invoke(app, ["agent", "skill"]).stdout.startswith("---\nname: gpu4agents\n")


def test_codex_gets_the_shared_skill_and_a_toml_server_entry(tmp_path, home, monkeypatch):
    codex = home / ".codex"
    codex.mkdir()
    config = codex / "config.toml"
    config.write_text('model = "gpt-5"\n\n[mcp_servers.gpu]\ncommand = "old"\nargs = []\n\n[mcp_servers.other]\ncommand = "x"\n')
    monkeypatch.setattr(install_mod.shutil, "which", lambda name: "/opt/bin/gpu" if name == "gpu" else None)

    result = install_mod.install(["codex"], tmp_path, global_scope=True)

    assert (home / ".agents" / "skills" / "gpu4agents" / "SKILL.md").is_file()
    text = config.read_text()
    assert 'model = "gpt-5"' in text
    assert '[mcp_servers.other]\ncommand = "x"' in text
    assert text.count("[mcp_servers.gpu]") == 1
    assert 'command = "/opt/bin/gpu"\nargs = ["mcp"]' in text
    assert "old" not in text
    assert {i["action"] for i in result["installed"]} == {"created", "updated"}
    again = install_mod.install(["codex"], tmp_path, global_scope=True)
    assert {i["action"] for i in again["installed"]} == {"unchanged"}


def test_codex_home_is_honored_and_a_missing_config_is_created(tmp_path, home, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    install_mod.install(["codex"], tmp_path, global_scope=True)
    text = (tmp_path / "codex-home" / "config.toml").read_text()
    assert text.startswith("[mcp_servers.gpu]\ncommand = ")


def test_gemini_project_and_global_settings(tmp_path, home):
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".gemini").mkdir()
    (project / ".gemini" / "settings.json").write_text(json.dumps({"theme": "dark"}))
    install_mod.install(["gemini"], project)
    data = json.loads((project / ".gemini" / "settings.json").read_text())
    assert data["theme"] == "dark"
    assert data["mcpServers"]["gpu"]["args"][-1] == "mcp"
    assert (project / ".agents" / "skills" / "gpu4agents" / "SKILL.md").is_file()

    install_mod.install(["gemini"], project, global_scope=True)
    assert json.loads((home / ".gemini" / "settings.json").read_text())["mcpServers"]["gpu"]


def test_auto_installs_for_the_runtimes_found(tmp_path, home, monkeypatch):
    (home / ".codex").mkdir()
    (home / ".gemini").mkdir()
    monkeypatch.setattr(install_mod.shutil, "which", lambda name: None)
    result = install_mod.install(["auto"], tmp_path, global_scope=True)
    assert result["targets"] == ["codex", "gemini"]
    nothing = tmp_path / "empty-home"
    nothing.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: nothing))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    empty = install_mod.install(["auto"], tmp_path, global_scope=True)
    assert empty["installed"] == [] and "no agent runtime found" in empty["notes"][0]


def test_the_old_gpu_skill_is_replaced_but_a_foreign_one_is_kept(tmp_path, home):
    skills = tmp_path / ".claude" / "skills"
    (skills / "gpu").mkdir(parents=True)
    (skills / "gpu" / "SKILL.md").write_text("---\nname: gpu\ndescription: x\n---\nuse gpu_setup\n")
    install_mod.install(["claude"], tmp_path)
    assert not (skills / "gpu").exists()

    (skills / "gpu").mkdir()
    (skills / "gpu" / "SKILL.md").write_text("---\nname: gpu\ndescription: someone else's\n---\n")
    install_mod.install(["claude"], tmp_path)
    assert (skills / "gpu" / "SKILL.md").is_file()


# -- the skill itself ---------------------------------------------------------


def test_skill_frontmatter_and_tool_names_match_the_server(tmp_path):
    text = install_mod.skill_text()
    front = yaml.safe_load(text.split("---")[1])
    assert front["name"] == "gpu4agents"
    assert 0 < len(front["description"]) <= 1024
    # gpu_providers/ is the adapter folder, not a tool
    mentioned = set(re.findall(r"\bgpu_[a-z]+\b", text)) - {"gpu_providers"}
    assert "gpu login" in text and "install.sh" in text
    assert mentioned <= set(Server(project=tmp_path).tools), mentioned - set(Server(project=tmp_path).tools)


@pytest.mark.skipif(sys.platform == "win32", reason="process groups")
def test_cancel_interrupts_the_command_and_records_its_exit(tmp_path):
    """SIGINT reaches the command (which cleans up) while the runner survives to record it."""
    job = Job(tmp_path)
    script = (
        "import sys,time\n"
        "try:\n    time.sleep(60)\n"
        "except KeyboardInterrupt:\n    print('cleaned up', flush=True); sys.exit(130)\n"
    )
    job.start([sys.executable, "-c", script], tmp_path)
    import time

    time.sleep(1.0)  # let the interpreter install its SIGINT handler
    assert job.cancel(timeout=20) is True
    report = job.report()
    assert report["status"] == "failed" and report["exit_code"] == 130
    assert job.out_path.read_text().strip() == "cleaned up"


def test_stop_cancels_a_running_setup_first(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "x")
    server = Server(project=tmp_path)
    _call(server, "gpu_init", model="org/m")
    job = server._job(tmp_path.resolve())
    job.start([sys.executable, "-c", "import time;time.sleep(60)"], tmp_path)
    error, payload = _call(server, "gpu_stop")
    assert not error, payload
    assert payload["setup_cancelled"]["status"] == "cancelled"
    assert not job.running()


def test_the_published_copies_of_the_skill_and_installer_match_the_sources():
    root = Path(__file__).resolve().parents[1]
    assert (root / "skills" / "gpu4agents" / "SKILL.md").read_text() == install_mod.skill_text()
    site = root / "site" / "web"
    if site.is_dir():
        assert (site / "install.sh").read_text() == (root / "install.sh").read_text()
        assert (site / "install.md").read_text() == (root / "INSTALL.md").read_text()
        assert (site / "providers.md").read_text() == (root / "PROVIDERS.md").read_text()
        assert (site / "jobs.md").read_text() == (root / "JOBS.md").read_text()
        assert (site / "serving.md").read_text() == (root / "SERVING.md").read_text()
        assert (site / "queue.md").read_text() == (root / "QUEUE.md").read_text()


def test_plugin_manifests_carry_the_package_version():
    from gpu4agents import __version__

    root = Path(__file__).resolve().parents[1]
    for manifest in (".claude-plugin/plugin.json", ".cursor-plugin/plugin.json"):
        data = json.loads((root / manifest).read_text())
        assert data["name"] == "gpu4agents"
        assert data["version"] == __version__, manifest
    market = json.loads((root / ".claude-plugin" / "marketplace.json").read_text())
    assert [p["name"] for p in market["plugins"]] == ["gpu4agents"]
