"""`gpu mcp` and `gpu agent` - let coding agents drive gpu."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.text import Text

from ..core.errors import JambuError
from ._common import console, get_context, handle, print_json

agent_app = typer.Typer(
    help="Give Claude Code, Codex, Gemini CLI, Cursor or Claude Desktop the gpu4agents skill and MCP tools.",
    no_args_is_help=True,
)


def mcp_command(
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Default project directory for the tools (default: $GPU_PROJECT_DIR, else the cwd).",
    ),
) -> None:
    """Run the local MCP server on stdio (started by the agent's client, not by hand)."""
    from ..agent.mcp import Server

    Server(project=project).serve()


@agent_app.command("install")
def install_command(
    ctx: typer.Context,
    target: List[str] = typer.Option(
        ["claude"],
        "--target",
        "-t",
        help="claude, codex, gemini, cursor, desktop (Claude Desktop), or auto (every runtime found). Repeat for several.",
    ),
    global_scope: bool = typer.Option(
        False, "--global", "-g", help="For every project of this user, not just this directory."
    ),
    directory: Path = typer.Option(
        Path("."), "--dir", help="Project directory (default: the current one)."
    ),
    sweep_timer: Optional[bool] = typer.Option(
        None,
        "--sweep-timer/--no-sweep-timer",
        help=(
            "Also schedule `gpu sweep --apply`, which removes machines left stopped (and billing) "
            "after stopped_ttl. Asked in a terminal when neither flag is given."
        ),
    ),
) -> None:
    """Install the gpu4agents skill and register the gpu MCP server."""
    import sys

    from ..agent.install import install, offer_sweep_timer

    context = get_context(ctx)
    try:
        result = install(list(target), directory, global_scope=global_scope)
    except JambuError as exc:
        handle(exc)
        return
    if sweep_timer is None and not context.json and sys.stdin.isatty():
        sweep_timer = typer.confirm(
            "A machine left stopped keeps billing for its disk on some providers (Vast.ai). "
            "Run a small timer every 5 minutes that removes it after stopped_ttl (24h by default)?",
            default=True,
        )
    result["sweep_timer"] = offer_sweep_timer(context, sweep_timer)
    if result["sweep_timer"].get("next"):
        result["notes"].append(result["sweep_timer"]["next"])
    if context.json:
        print_json(result)
        return
    for item in result["installed"]:
        style = "dim" if item["action"] == "unchanged" else "green"
        console.print(Text(f"+ {item['what']}: {item['path']} ({item['action']})", style=style))
    for note in result["notes"]:
        console.print(Text(f"  {note}", style="cyan"))


@agent_app.command("skill")
def skill_command() -> None:
    """Print the bundled SKILL.md (to copy into another agent's skill directory)."""
    from ..agent.install import skill_text

    print(skill_text(), end="")
