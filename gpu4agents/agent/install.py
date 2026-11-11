"""`gpu agent install`: put the skill and the MCP server where agents look for them.

Skills follow the open Agent Skills layout, one folder with a SKILL.md:

- claude:              ``.claude/skills/gpu4agents/`` (Claude Code; Cursor reads it too)
- codex, gemini, cursor: ``.agents/skills/gpu4agents/``, the shared directory
  Codex, Gemini CLI and Cursor all read

Each is per project, or under the home directory with --global. The MCP server
is registered per client:

- claude:  ``.mcp.json`` in the project; with --global, ``claude mcp add -s user``
- codex:   ``[mcp_servers.gpu]`` in ``$CODEX_HOME/config.toml`` (``~/.codex``)
- gemini:  ``mcpServers.gpu`` in ``.gemini/settings.json``, or ``~/.gemini/settings.json``
- cursor:  ``.cursor/mcp.json`` in the project, or ``~/.cursor/mcp.json``
- desktop: Claude Desktop's ``claude_desktop_config.json``, pinned to this
  project, plus a zip of the skill to upload in Settings

``auto`` picks every runtime found on this machine. Existing config files are
merged, never replaced: only the ``gpu`` server entry is written.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

from ..core.errors import ConfigError
from ..core.paths import gpu_home

SERVER_NAME = "gpu"
SKILL_NAME = "gpu4agents"
LEGACY_SKILL_NAME = "gpu"
TARGETS = ("claude", "codex", "gemini", "cursor", "desktop")
SHARED_SKILL_TARGETS = ("codex", "gemini", "cursor")


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


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def detect_targets() -> list[str]:
    """Runtimes present on this machine: their command on PATH or their config directory."""
    home = Path.home()
    found = []
    if shutil.which("claude") or (home / ".claude").is_dir():
        found.append("claude")
    if shutil.which("codex") or codex_home().is_dir():
        found.append("codex")
    if shutil.which("gemini") or (home / ".gemini").is_dir():
        found.append("gemini")
    if shutil.which("cursor") or shutil.which("cursor-agent") or (home / ".cursor").is_dir():
        found.append("cursor")
    return found


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


_TOML_TABLE = re.compile(r"^\s*\[\[?[^\]]+\]\]?\s*(#.*)?$")


def merge_codex_server(path: Path, command: list[str]) -> str:
    """Write ``[mcp_servers.gpu]`` into Codex's config.toml, keeping everything else.

    Python 3.10 has no TOML parser, and only one table is ours, so the table is
    replaced as text: from its header to the next table header.
    """
    header = f"[mcp_servers.{SERVER_NAME}]"
    block = [
        header,
        f"command = {json.dumps(command[0])}",
        f"args = {json.dumps([*command[1:], 'mcp'])}",
    ]
    lines = path.read_text().splitlines() if path.is_file() else []
    start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
    if start is None:
        while lines and not lines[-1].strip():
            lines.pop()
        new = lines + ([""] if lines else []) + block
        action = "created"
    else:
        end = next(
            (i for i in range(start + 1, len(lines)) if _TOML_TABLE.match(lines[i])),
            len(lines),
        )
        old = [line for line in lines[start:end] if line.strip()]
        if old == block:
            return "unchanged"
        tail = lines[end:]
        new = lines[:start] + block + ([""] + tail if tail else [])
        action = "updated"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(new) + "\n")
    return action


def _is_our_legacy_skill(path: Path) -> bool:
    """The skill shipped before 0.6.0 under the name ``gpu``."""
    try:
        text = path.read_text()
    except OSError:
        return False
    return text.startswith(f"---\nname: {LEGACY_SKILL_NAME}\n") and "gpu_setup" in text


def write_skill(directory: Path) -> str:
    target = directory / SKILL_NAME / "SKILL.md"
    legacy = directory / LEGACY_SKILL_NAME / "SKILL.md"
    if _is_our_legacy_skill(legacy):
        legacy.unlink()
        if not any(legacy.parent.iterdir()):
            legacy.parent.rmdir()
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


def resolve_targets(targets: list[str]) -> list[str]:
    wanted: list[str] = []
    for target in targets:
        expanded = detect_targets() if target == "auto" else [target]
        for name in expanded:
            if name not in wanted:
                wanted.append(name)
    unknown = [t for t in wanted if t not in TARGETS]
    if unknown:
        raise ConfigError(
            f"unknown target {', '.join(unknown)} (choose from {', '.join(TARGETS)}, or auto)"
        )
    return wanted


def install(
    targets: list[str],
    project_dir: Path,
    global_scope: bool = False,
    run=subprocess.run,
) -> dict[str, Any]:
    """Install for each target. ``run`` is a seam for tests (``claude mcp add``)."""
    targets = resolve_targets(targets)
    project_dir = project_dir.resolve()
    base = Path.home() if global_scope else project_dir
    done: list[dict[str, Any]] = []
    notes: list[str] = []
    if not targets:
        notes.append(
            "no agent runtime found (Claude Code, Codex, Gemini CLI, Cursor); "
            "name one with -t claude|codex|gemini|cursor|desktop"
        )

    if "claude" in targets:
        skills = base / ".claude" / "skills"
        done.append({"what": "skill (Claude Code)", "path": str(skills / SKILL_NAME / "SKILL.md"), "action": write_skill(skills)})

    shared = [t for t in targets if t in SHARED_SKILL_TARGETS]
    if shared:
        skills = base / ".agents" / "skills"
        names = {"codex": "Codex", "gemini": "Gemini CLI", "cursor": "Cursor"}
        label = ", ".join(names[t] for t in shared)
        done.append({"what": f"skill ({label})", "path": str(skills / SKILL_NAME / "SKILL.md"), "action": write_skill(skills)})

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

    if "codex" in targets:
        # Codex reads MCP servers from the user config; the server works in
        # whatever directory Codex runs, so one entry serves every project.
        path = codex_home() / "config.toml"
        done.append({"what": "MCP server (Codex)", "path": str(path), "action": merge_codex_server(path, gpu_command(portable=False))})

    if "gemini" in targets:
        command = gpu_command(portable=not global_scope)
        entry = {"command": command[0], "args": [*command[1:], "mcp"]}
        path = base / ".gemini" / "settings.json"
        done.append({"what": "MCP server (Gemini CLI)", "path": str(path), "action": merge_server(path, entry)})

    if "cursor" in targets:
        command = gpu_command(portable=not global_scope)
        entry = {
            "type": "stdio",
            "command": command[0],
            "args": [*command[1:], "mcp", "--project", "${workspaceFolder}"],
        }
        path = base / ".cursor" / "mcp.json"
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
            "Claude Desktop does not get your shell's environment: keep the provider key in "
            "~/.gpu/credentials (gpu login) or the project .env"
        )

    if any(t in targets for t in ("claude", "codex", "gemini", "cursor")):
        notes.append("restart the agent (or start a new session) so it loads the skill and the gpu tools")
    return {"installed": done, "targets": targets, "notes": notes}


def _enable_timer(result: dict[str, Any]) -> bool:
    """Run the enable command `gpu sweep --install-timer` prints (systemctl or launchctl)."""
    proc = subprocess.run(["sh", "-c", result["enable"]], capture_output=True, text=True)
    return proc.returncode == 0


def offer_sweep_timer(context: Any, wanted: Optional[bool]) -> dict[str, Any]:
    """The install flow's one question: schedule the sweep, which enforces stopped_ttl.

    Nothing on a stopped machine can remove it, so where a stop still bills
    (Vast.ai) only something on this computer can: the sweep timer.
    """
    from ..cli import timer as timer_mod
    from ..cli.account import _require_file_credentials

    howto = "`gpu sweep --install-timer` schedules the cleanup of machines left stopped (they bill for disk on Vast.ai)"
    if timer_mod.is_installed():
        return {"installed": True, "action": "unchanged"}
    if not wanted:
        return {"installed": False, "action": "skipped", "next": howto}
    try:
        _require_file_credentials(context)
        result = timer_mod.install_timer(300.0)
    except (ConfigError, OSError) as exc:
        return {"installed": False, "action": "failed", "error": str(exc), "next": howto}
    enabled = _enable_timer(result)
    out = {"installed": enabled, "action": "created" if enabled else "written", "files": result["files"]}
    out["timer"] = result.get("timer") or result.get("plist")
    if not enabled:
        out["next"] = f"enable it with: {result['enable']}"
    return out
