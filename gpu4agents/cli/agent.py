"""`gpu mcp` and `gpu agent` - let coding agents drive gpu."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.text import Text

from ..core.errors import JambuError
from ._common import console, get_context, handle, print_json

agent_app = typer.Typer(
    help="Give Claude Code, Cursor or Claude Desktop the gpu skill and MCP tools.",
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
        help="claude (Claude Code), cursor, desktop (Claude Desktop). Repeat for several.",
    ),
    global_scope: bool = typer.Option(
        False, "--global", "-g", help="For every project of this user, not just this directory."
    ),
    directory: Path = typer.Option(
        Path("."), "--dir", help="Project directory (default: the current one)."
    ),
) -> None:
    """Install the gpu skill and register the gpu MCP server."""
    from ..agent.install import install

    context = get_context(ctx)
    try:
        result = install(list(target), directory, global_scope=global_scope)
    except JambuError as exc:
        handle(exc)
        return
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
