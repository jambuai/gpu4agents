"""`gpu agent install`: put the skill and the MCP server where agents look for them.

One skill copy, in ``.claude/skills/gpu/`` (or ``~/.claude/skills/gpu/``),
serves both Claude Code and Cursor, which also reads Claude's skill
directories. The MCP server is registered per client:

- claude:  ``.mcp.json`` in the project; with --global, ``claude mcp add -s user``
- cursor:  ``.cursor/mcp.json`` in the project, or ``~/.cursor/mcp.json``
- desktop: Claude Desktop's ``claude_desktop_config.json``, pinned to this
  project, plus a zip of the skill to upload in Settings

Existing config files are merged, never replaced: only the ``gpu`` server
entry is written.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

from ..core.errors import ConfigError
from ..core.paths import gpu_home

SERVER_NAME = "gpu"
SKILL_NAME = "gpu"
TARGETS = ("claude", "cursor", "desktop")


def skill_source() -> Path:
    return Path(__file__).parent / "skill" / "SKILL.md"


def skill_text() -> str:
    return skill_source().read_text()


def gpu_command(portable: bool) -> list[str]:
    """How a client starts `gpu`.

    A project file that may be committed says plain ``gpu`` when it is on
    PATH; anything machine-local uses an absolute path, because GUI apps
    (Claude Desktop, Cursor) do not get the shell's PATH.
    """
    found = shutil.which("gpu")
    if found and portable:
        return ["gpu"]
    if found:
        return [str(Path(found).resolve())]
    return [sys.executable, "-m", "gpu4agents"]


def desktop_config_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Claude" / "claude_desktop_config.json"
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "Claude" / "claude_desktop_config.json"


def merge_server(path: Path, entry: dict[str, Any]) -> str:
    """Write ``mcpServers.gpu`` into a JSON config. Returns created/updated/unchanged."""
    data: dict[str, Any] = {}
    if path.is_file():
        text = path.read_text().strip()
        if text:
            try:
                data = json.loads(text)
            except ValueError as exc:
                raise ConfigError(f"{path} is not valid JSON ({exc}); fix it or add the gpu server by hand") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path} does not hold a JSON object")
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ConfigError(f"{path}: mcpServers is not an object")
    if servers.get(SERVER_NAME) == entry:
        return "unchanged"
    action = "updated" if SERVER_NAME in servers else "created"
    servers[SERVER_NAME] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return action


def write_skill(directory: Path) -> str:
    target = directory / SKILL_NAME / "SKILL.md"
    text = skill_text()
    if target.is_file() and target.read_text() == text:
        return "unchanged"
    action = "updated" if target.exists() else "created"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return action


def write_skill_zip(path: Path) -> str:
    """The skill as Claude Desktop / claude.ai expect an upload: a folder in a zip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{SKILL_NAME}/SKILL.md", skill_text())
    return "created"


def install(
    targets: list[str],
    project_dir: Path,
    global_scope: bool = False,
    run=subprocess.run,
) -> dict[str, Any]:
    """Install for each target. ``run`` is a seam for tests (``claude mcp add``)."""
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise ConfigError(f"unknown target {', '.join(unknown)} (choose from {', '.join(TARGETS)})")
    project_dir = project_dir.resolve()
    done: list[dict[str, Any]] = []
    notes: list[str] = []

    if "claude" in targets or "cursor" in targets:
        skills = (Path.home() if global_scope else project_dir) / ".claude" / "skills"
        done.append({"what": "skill (Claude Code, Cursor)", "path": str(skills / SKILL_NAME / "SKILL.md"), "action": write_skill(skills)})

    if "claude" in targets:
        if global_scope:
            command = gpu_command(portable=False)
            argv = ["claude", "mcp", "add", "--scope", "user", SERVER_NAME, "--", *command, "mcp"]
            if shutil.which("claude"):
                # `add` refuses an existing name; replace it so a reinstall picks up a new path.
                run(["claude", "mcp", "remove", "--scope", "user", SERVER_NAME], capture_output=True, text=True)
                proc = run(argv, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise ConfigError(f"`{' '.join(argv)}` failed: {(proc.stderr or proc.stdout).strip()}")
                done.append({"what": "MCP server (Claude Code, user scope)", "path": "claude mcp", "action": "created"})
            else:
                notes.append(f"claude CLI not found; register the server with: {' '.join(argv)}")
        else:
            command = gpu_command(portable=True)
            entry = {"command": command[0], "args": [*command[1:], "mcp"]}
            path = project_dir / ".mcp.json"
            done.append({"what": "MCP server (Claude Code)", "path": str(path), "action": merge_server(path, entry)})
            notes.append("Claude Code asks once to approve the project's .mcp.json servers")

    if "cursor" in targets:
        command = gpu_command(portable=not global_scope)
        entry = {
            "type": "stdio",
            "command": command[0],
            "args": [*command[1:], "mcp", "--project", "${workspaceFolder}"],
        }
        path = (Path.home() if global_scope else project_dir) / ".cursor" / "mcp.json"
        done.append({"what": "MCP server (Cursor)", "path": str(path), "action": merge_server(path, entry)})

    if "desktop" in targets:
        command = gpu_command(portable=False)
        entry = {"command": command[0], "args": [*command[1:], "mcp", "--project", str(project_dir)]}
        path = desktop_config_path()
        done.append({"what": "MCP server (Claude Desktop)", "path": str(path), "action": merge_server(path, entry)})
        archive = gpu_home() / "gpu-skill.zip"
        done.append({"what": "skill zip (Claude Desktop)", "path": str(archive), "action": write_skill_zip(archive)})
        notes.append(f"Claude Desktop: restart it, then upload {archive} under Settings > Capabilities > Skills")
        notes.append(
            "Claude Desktop does not get your shell's environment: keep VAST_API_KEY in the "
            "project .env or ~/.gpu/credentials"
        )

    return {"installed": done, "notes": notes}
