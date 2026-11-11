"""`gpu lock|unlock|stop|destroy|guard` - lifecycle control."""

from __future__ import annotations

import typer
from rich.text import Text

from ..core.durations import parse_duration
from ..core.errors import ConfigError, JambuError
from ..core.state import to_iso
from ._common import console, get_context, handle, print_json


def lock_command(
    ctx: typer.Context,
    duration: str = typer.Option(
        "", "--for", help="Lock TTL, e.g. 2h. Omit to use lifecycle.lock.default_ttl."
    ),
    reason: str = typer.Option("", "--reason", help="Why the instance must stay up."),
) -> None:
    """Prevent automatic shutdown until the lock expires (spec section 11)."""
    context = get_context(ctx)
    try:
        engine = context.engine()
        if duration:
            ttl = parse_duration(duration)
        else:
            ttl = engine.config.lock_default_ttl_s
        result = engine.lock(ttl, reason=reason)
    except ValueError as exc:
        handle(ConfigError(str(exc)))
        return
    except JambuError as exc:
        handle(exc)
        return

    if context.json:
        print_json(result)
        return
    until = result.get("locked_until")
    console.print(
        Text(
            f"+ locked until {to_iso(until)}" if until else "+ locked indefinitely",
            style="yellow",
        )
    )


def unlock_command(ctx: typer.Context) -> None:
    """Resume normal lifecycle rules."""
    context = get_context(ctx)
    try:
        result = context.engine().unlock()
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json(result)
        return
    console.print(Text("+ unlocked; idle and lifetime rules resume", style="green"))


def stop_command(
    ctx: typer.Context,
    reason: str = typer.Option("manual", "--reason", help="Recorded with the stop event."),
) -> None:
    """Stop the compute resource (idempotent)."""
    context = get_context(ctx)
    try:
        result = context.engine().stop(reason=reason)
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json(result)
        return
    if result.get("warning"):
        console.print(Text(f"! {result['warning']}", style="yellow"))


def destroy_command(
    ctx: typer.Context,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    reason: str = typer.Option("manual", "--reason", help="Recorded with the destroy event."),
    discard_outputs: bool = typer.Option(
        False,
        "--discard-outputs",
        help="Destroy even if job results could not be downloaded first (they are deleted).",
    ),
) -> None:
    """Destroy the remote resource (idempotent). Job results are downloaded first."""
    context = get_context(ctx)
    try:
        engine = context.engine()
        if not yes and not context.json:
            state = engine.store.load()
            if state.instance_id:
                confirm = typer.confirm(
                    f"destroy instance {state.instance_id} on {engine.provider.display_name}?"
                )
                if not confirm:
                    console.print("aborted")
                    raise typer.Exit(1)
        result = engine.destroy(reason=reason, discard_outputs=discard_outputs)
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json(result)


def guard_command(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Evaluate the policy without enforcing it."
    ),
) -> None:
    """Evaluate the lifecycle rule now (external controller path, spec section 14).

    The remote watchdog already does this on the instance. Use this from cron
    or CI as a second, independent enforcement path.
    """
    context = get_context(ctx)
    try:
        result = context.engine().guard_tick(apply=not dry_run)
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json(result)
        return
    action = result.get("action")
    reason = result.get("reason")
    style = "yellow" if action != "none" else "green"
    console.print(Text(f"{action}: {reason}", style=style))


def adopt_command(
    ctx: typer.Context,
    instance_id: str = typer.Argument(..., help="The orphan's id, as `gpu ls` shows it."),
) -> None:
    """Take over a machine gpu rented whose project is gone (`gpu ls` shows it as orphan).

    Run it in the project that should own it. Where the provider keeps the
    machine's boot script (Vast.ai), its watchdog and model key come back and
    `gpu setup` reuses it if this gpu.yaml matches; otherwise it can still be
    watched, stopped and destroyed from here.
    """
    context = get_context(ctx)
    try:
        result = context.engine().adopt(instance_id)
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json(result)
        return
    console.print(
        Text(
            f"+ adopted {result['instance_id']} ({result['gpu']}, {result['state']})",
            style="green",
        )
    )
    console.print(f"next: {result['next']}")
