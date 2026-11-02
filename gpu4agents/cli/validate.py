"""`gpu validate` - fail before provisioning (spec sections 5, 15)."""

from __future__ import annotations

import typer
from rich.text import Text

from ..core.errors import JambuError
from ._common import console, get_context, handle, print_json


def validate_command(
    ctx: typer.Context,
    offline: bool = typer.Option(
        False, "--offline", help="Skip provider API checks (schema and policy only)."
    ),
) -> None:
    """Validate jambu.yaml, credentials, capabilities and available capacity."""
    context = get_context(ctx)
    try:
        engine = context.engine()
        result = engine.validate(check_provider=not offline)
    except JambuError as exc:
        if context.json:
            print_json({"ok": False, "errors": [{"message": str(exc)}]})
            raise typer.Exit(getattr(exc, "exit_code", 1)) from exc
        handle(exc)
        return

    payload = {
        "ok": result.ok,
        "provider": engine.provider.name,
        "model": engine.config.model.id,
        "runtime": engine.config.runtime.engine,
        "fingerprint": engine.config.fingerprint(),
        "errors": [{"field": i.field, "message": i.message} for i in result.errors],
        "warnings": [{"field": i.field, "message": i.message} for i in result.warnings],
    }
    if context.json:
        print_json(payload)
        raise typer.Exit(0 if result.ok else 2)

    console.print(
        f"config: [bold]{engine.config.source_path}[/bold]\n"
        f"provider: [bold]{engine.provider.display_name}[/bold]  "
        f"runtime: [bold]{engine.runtime.describe()}[/bold]"
    )
    for issue in result.warnings:
        console.print(Text(f"! {issue.field or 'config'}: {issue.message}", style="yellow"))
    for issue in result.errors:
        console.print(Text(f"x {issue.field or 'config'}: {issue.message}", style="bold red"))

    if result.ok:
        console.print(Text("+ configuration is valid", style="bold green"))
    else:
        console.print(Text(f"{len(result.errors)} error(s) must be fixed", style="bold red"))
        raise typer.Exit(2)
