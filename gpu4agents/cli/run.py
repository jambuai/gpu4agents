"""`gpu run` - execute a workload against a guaranteed runtime (spec section 16)."""

from __future__ import annotations

from typing import List

import typer

from ..core.errors import JambuError, WorkloadFailed
from ._common import get_context, handle, print_json, register_project


def run_command(
    ctx: typer.Context,
    command: List[str] = typer.Argument(
        ...,
        help=(
            "Command to execute, e.g. `python experiments/run.py`. Flags are passed "
            "through; put `--` first if one collides with a gpu option."
        ),
    ),
    label: str = typer.Option("", "--label", help="Human label recorded with the workload."),
) -> None:
    """Ensure the runtime is READY, then run the command with heartbeat tracking."""
    context = get_context(ctx)
    try:
        engine = context.engine()
        register_project(context)  # `run` provisions when nothing is up
        engine.run(list(command), label=label)
    except WorkloadFailed as exc:
        if context.json:
            print_json({"ok": False, "returncode": exc.returncode})
        raise typer.Exit(exc.returncode) from exc
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json({"ok": True, "returncode": 0})
