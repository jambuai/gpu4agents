"""`gpu status` / `logs` / `events` - reconciled observability."""

from __future__ import annotations

import time

import typer
from rich.table import Table
from rich.text import Text

from ..core.durations import format_duration
from ..core.errors import JambuError
from ._common import (
    console,
    fmt_money,
    fmt_time,
    get_context,
    handle,
    kv_table,
    print_json,
    state_text,
)


def status_command(
    ctx: typer.Context,
    watch: float = typer.Option(
        0, "--watch", "-w", help="Refresh every N seconds (0 = once)."
    ),
) -> None:
    """Reconcile local and provider state, then report the normalized result."""
    context = get_context(ctx)
    while True:
        try:
            payload = context.engine().status()
        except JambuError as exc:
            handle(exc)
            return

        if context.json:
            print_json(payload)
            return

        _render(payload)
        if not watch:
            return
        time.sleep(max(2.0, watch))
        console.rule(style="dim")


def _render(payload: dict) -> None:
    decision = payload.get("decision") or {}
    watchdog = payload.get("watchdog") or {}
    rows = [
        ("state", state_text(payload.get("state", "unknown"))),
        ("provider", payload.get("provider")),
        ("instance", payload.get("instance_id") or "-"),
        ("model", payload.get("model")),
        ("endpoint", payload.get("endpoint") or "-"),
        ("created", fmt_time(payload.get("created_at"))),
        (
            "age",
            format_duration(payload.get("age_seconds"))
            if payload.get("age_seconds") is not None
            else None,
        ),
        ("idle", format_duration(payload.get("idle_seconds"))),
        ("cost", _cost(payload)),
        ("locked", _lock(payload)),
        (
            "watchdog",
            _watchdog(watchdog),
        ),
        ("guard", _guard(decision, payload)),
    ]
    console.print(kv_table("Runtime", rows))

    workloads = [w for w in payload.get("workloads", []) if w["state"] in ("running", "queued")]
    recent = sorted(
        payload.get("workloads", []), key=lambda w: w.get("started_at") or 0, reverse=True
    )[:5]
    if recent:
        table = Table(title="Workloads", title_justify="left", box=None)
        for column in ("ID", "STATE", "LABEL", "STARTED", "RC"):
            table.add_column(column)
        for record in recent:
            table.add_row(
                record["id"],
                Text(
                    record["state"],
                    style="cyan" if record["state"] == "running" else "dim",
                ),
                (record.get("label") or "")[:48],
                fmt_time(record.get("started_at")) or "-",
                "-" if record.get("returncode") is None else str(record["returncode"]),
            )
        console.print(table)
    if workloads:
        console.print(
            Text(f"  {len(workloads)} active workload(s) hold the instance open", style="cyan")
        )


def _cost(payload: dict) -> str:
    hourly = payload.get("hourly_cost_usd")
    session = payload.get("session_cost_usd")
    if hourly is None and session is None:
        return "-"
    parts = []
    if hourly is not None:
        parts.append(f"{fmt_money(hourly)}/h")
    if session:
        parts.append(f"{fmt_money(session)} this session")
    return "  ".join(parts) or "-"


def _lock(payload: dict) -> Text:
    if not payload.get("locked"):
        return Text("no", style="dim")
    until = fmt_time(payload.get("locked_until"))
    return Text(f"yes (until {until})" if until else "yes (indefinite)", style="yellow")


def _watchdog(watchdog: dict) -> Text:
    if not watchdog.get("enabled"):
        return Text("disabled - shutdown depends on this CLI staying alive", style="yellow")
    if watchdog.get("reachable"):
        return Text(f"live at {watchdog.get('url')}", style="green")
    if watchdog.get("installed"):
        return Text(
            f"unreachable ({watchdog.get('last_error') or 'no answer'})", style="yellow"
        )
    return Text("not installed", style="yellow")


def _guard(decision: dict, payload: dict) -> Text:
    action = decision.get("action", "none")
    reason = decision.get("reason") or "-"
    if action == "none":
        policy = payload.get("policy") or {}
        idle_timeout = policy.get("idle_timeout_s")
        idle = payload.get("idle_seconds") or 0
        if idle_timeout and reason == "within_policy":
            remaining = max(0.0, float(idle_timeout) - float(idle))
            return Text(
                f"ok ({reason}); auto-stop in {format_duration(remaining)} if idle",
                style="green",
            )
        if (decision.get("detail") or {}).get("idle_skipped"):
            return Text(f"ok ({reason}); idle left to the unreachable watchdog", style="yellow")
        return Text(f"ok ({reason})", style="green")
    return Text(f"{action} pending: {reason}", style="bold yellow")


def logs_command(
    ctx: typer.Context,
    lines: int = typer.Option(200, "--lines", "-n", help="How many lines to show."),
    source: str = typer.Option(
        "",
        "--source",
        "-s",
        help="runtime | onstart | watchdog (read on the instance through the "
        "watchdog) or provider (container log). Default: runtime when the "
        "watchdog answers, else provider.",
    ),
) -> None:
    """Show the instance's logs: the model server's by default."""
    context = get_context(ctx)
    try:
        used, output = context.engine().read_logs(limit=lines, source=source or None)
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json({"source": used, "lines": output})
        return
    for line in output:
        console.print(line, markup=False, highlight=False)


def events_command(
    ctx: typer.Context,
    limit: int = typer.Option(40, "--limit", "-n", help="How many events to show."),
    event: str = typer.Option("", "--event", help="Filter by event name."),
) -> None:
    """Show the structured lifecycle event log (spec section 24)."""
    context = get_context(ctx)
    try:
        engine = context.engine()
    except JambuError as exc:
        handle(exc)
        return
    records = engine.events.tail(limit=limit, event=event or None)
    if context.json:
        print_json({"events": records})
        return
    if not records:
        console.print(Text("no events recorded yet", style="dim"))
        return
    table = Table(title="Lifecycle events", title_justify="left", box=None)
    for column in ("TS", "EVENT", "DETAIL"):
        table.add_column(column)
    for record in records:
        detail = {
            k: v
            for k, v in record.items()
            if k not in ("ts", "event", "pid", "provider", "source")
        }
        table.add_row(
            str(record.get("ts", "")),
            Text(str(record.get("event", "")), style=_event_style(record.get("event", ""))),
            ", ".join(f"{k}={v}" for k, v in detail.items())[:110],
        )
    console.print(table)


def _event_style(event: str) -> str:
    if "error" in event or "failed" in event:
        return "red"
    if "guard" in event or "stopped" in event or "destroyed" in event:
        return "yellow"
    if "ready" in event or "created" in event:
        return "green"
    return "white"
