"""Shared CLI plumbing: config loading, output and error rendering."""

from __future__ import annotations

import json as jsonlib
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..core.config import RuntimeConfig, load_config
from ..core.engine import Engine
from ..core.errors import JambuError
from ..core.state import to_iso

console = Console()
err_console = Console(stderr=True)

STATE_STYLES = {
    "ready": "bold green",
    "busy": "bold cyan",
    "idle": "yellow",
    "starting": "cyan",
    "provisioning": "cyan",
    "stopped": "dim",
    "destroyed": "dim",
    "failed": "bold red",
    "unknown": "magenta",
    "none": "dim",
}


class Context:
    """Per-invocation state carried on the typer context object."""

    def __init__(self) -> None:
        self.config_path: Optional[Path] = None
        self.json: bool = False
        self.verbose: bool = False
        self._config: Optional[RuntimeConfig] = None
        self._engine: Optional[Engine] = None

    def config(self) -> RuntimeConfig:
        if self._config is None:
            self._config = load_config(self.config_path)
        return self._config

    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = Engine(self.config(), echo=self.echo)
        return self._engine

    def echo(self, message: str) -> None:
        if self.json:
            return
        style = ""
        if message.startswith("!"):
            style = "yellow"
        elif message.startswith("+"):
            style = "green"
        elif message.startswith("-"):
            style = "red"
        elif message.startswith(">"):
            style = "cyan"
        console.print(Text(message, style=style) if style else message)


def get_context(ctx: typer.Context) -> Context:
    if ctx.obj is None:
        ctx.obj = Context()
    return ctx.obj


def fail(message: str, code: int = 1) -> None:
    err_console.print(Text(f"error: {message}", style="bold red"))
    raise typer.Exit(code)


def handle(exc: Exception) -> None:
    """Render a JambuError the same way regardless of which provider raised it."""
    if isinstance(exc, JambuError):
        err_console.print(Text(f"error: {exc}", style="bold red"))
        raise typer.Exit(getattr(exc, "exit_code", 1))
    raise exc


def print_json(payload: Any) -> None:
    sys.stdout.write(jsonlib.dumps(payload, indent=2, default=str) + "\n")


def state_text(state: str) -> Text:
    return Text(state.upper(), style=STATE_STYLES.get(state, "white"))


def kv_table(title: str, rows: list[tuple[str, Any]]) -> Table:
    table = Table(title=title, show_header=False, box=None, title_justify="left", pad_edge=False)
    table.add_column("field", style="dim", no_wrap=True)
    table.add_column("value")
    for key, value in rows:
        if value is None:
            continue
        table.add_row(key, value if isinstance(value, Text) else str(value))
    return table


def fmt_time(ts: Optional[float]) -> Optional[str]:
    return to_iso(ts) if ts else None


def fmt_money(value: Optional[float], suffix: str = "") -> Optional[str]:
    if value is None:
        return None
    return f"${value:.3f}{suffix}"


def register_project(context: Context) -> None:
    """Put this project in $GPU_HOME/projects.json so `gpu sweep` guards it.

    Done before provisioning: a setup that dies halfway must still leave an
    instance the sweep judges by this project's policy, not as an orphan.
    """
    from ..core.sweep import register_project as _register

    try:
        _register(context.config())
    except OSError as exc:
        context.echo(f"! could not register the project for `gpu sweep`: {exc}")
