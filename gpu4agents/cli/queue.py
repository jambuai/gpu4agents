"""`gpu queue` - submit jobs to the queue, follow them, and run or control the coordinator."""

from __future__ import annotations

import functools
import re
import signal
import threading
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from rich.text import Text

from ..core.errors import ConfigError, JambuError
from ..queue import api
from . import _common
from ._common import Context, console, fmt_money, get_context, handle, print_json, state_text
from .job import _parse_env

queue_app = typer.Typer(
    help=(
        "Queue mode: submit job directories and get their outputs back; a coordinator "
        "(`gpu queue run`) runs them on a pool of machines. Needs `queue:` in gpu.yaml."
    ),
    no_args_is_help=True,
)


_URL_PASSWORD = re.compile(r"(://[^:/@\s]*):[^@\s]*@")


def _guard(fn):
    """A queue backend or output store that cannot be reached is one clean error line (exit 4),
    never a traceback, and never a password."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except JambuError:
            raise
        except Exception as exc:  # noqa: BLE001
            root = type(exc).__module__.split(".")[0]
            if not (isinstance(exc, (OSError, TimeoutError)) or root in ("redis", "botocore", "boto3", "urllib3")):
                raise
            text = _URL_PASSWORD.sub(r"\1:***@", str(exc))[:200]
            _common.fail(f"the queue or its output store is unreachable ({type(exc).__name__}: {text})", 4)

    return wrapper


@queue_app.callback()
def queue_callback(ctx: typer.Context) -> None:
    # The main callback leaves "queue" quiet; only `run` stays so.
    if ctx.invoked_subcommand != "run":
        _common.warn_stopped_billing()


def _project(context: Context) -> Optional[Path]:
    return context.config_path


def _emit(context: Context, payload, human) -> None:
    if context.json:
        print_json(payload)
    else:
        human(payload)


@queue_app.command("submit")
@_guard
def submit_command(
    ctx: typer.Context,
    directory: Path = typer.Argument(Path("."), help="Directory of code (gpujob.yaml with `run:`)."),
    key: Optional[str] = typer.Option(None, "--key", help="Idempotency key: the same key never queues twice."),
    env: Optional[List[str]] = typer.Option(None, "--env", "-e", help="KEY=VALUE for the job (repeatable)."),
) -> None:
    """Pack DIR and queue it. Prints the job id at once; the job runs when a coordinator has a machine."""
    context = get_context(ctx)
    try:
        result = api.submit_job(directory, key=key, project=_project(context), env=_parse_env(env))
    except JambuError as exc:
        handle(exc)
        return

    def human(r: dict) -> None:
        console.print(Text(f"+ queued {r['job_id']}" + (" (already queued under this key)" if r["existing"] else ""), style="green"))
        if r["paused"]:
            console.print(Text("! the queue is paused: it starts after `gpu queue resume`", style="yellow"))
        console.print(f"next: gpu queue status {r['job_id']}")

    _emit(context, result, human)


def _job_human(info: dict) -> None:
    console.print(
        f"{info['id']}  ",
        state_text(info["state"]),
        f"  attempts {len(info['attempts'])}" + (f"  machine {info['machine']}" if info.get("machine") else ""),
    )
    if info.get("cancel_requested") and not info.get("final"):
        console.print("  cancel requested")
    result = info.get("result") or {}
    if result.get("reason"):
        console.print(f"  {result['reason']}")
    if info.get("outputs_ref"):
        console.print(f"  outputs: {info['outputs_ref'].get('location')}  (gpu queue pull {info['id']})")


def _summary_human(s: dict) -> None:
    by_state = s["queue"]["by_state"]
    counts = ", ".join(f"{n} {k}" for k, n in sorted(by_state.items()) if n) or "empty"
    console.print(f"queue: {counts}" + ("  PAUSED" if s["paused"] else "") + ("  drain requested" if s["drain_requested"] else ""))
    lease = s["coordinator"]
    console.print(
        f"coordinator: {lease['owner']} (lease {lease['lease_remaining_s']:.0f}s)" if lease
        else "coordinator: none holds the lease (start one with `gpu queue run`)"
    )
    pool = s["pool"]
    if pool.get("machines") is None:
        console.print(f"pool: unavailable ({pool.get('error')})")
        return
    console.print(f"pool: {len(pool['machines'])} machines, {fmt_money(pool['usd_per_hour'], '/h')}")
    if pool["machines"]:
        table = Table(box=None, pad_edge=False)
        for col in ("slot", "state", "$/h", "jobs"):
            table.add_column(col)
        for m in pool["machines"]:
            table.add_row(m["slot"], m["state"], f"{m['hourly_usd']:.3f}", ",".join(m["jobs"]) or "-")
        console.print(table)


@queue_app.command("status")
@_guard
def status_command(
    ctx: typer.Context, job: Optional[str] = typer.Argument(None, help="Job id; omit for the whole queue."),
) -> None:
    """One job's state, or (no JOB) the queue: stats, pool machines and their cost, coordinator, paused."""
    context = get_context(ctx)
    try:
        payload = api.status(job, _project(context)) if job else api.summary(_project(context))
    except JambuError as exc:
        handle(exc)
        return
    _emit(context, payload, _job_human if job else _summary_human)


@queue_app.command("ls")
@_guard
def ls_command(
    ctx: typer.Context,
    state: Optional[str] = typer.Option(None, "--state", help="Only jobs in this state."),
) -> None:
    """List jobs, oldest first."""
    context = get_context(ctx)
    try:
        rows = api.listing(state, _project(context))
    except JambuError as exc:
        handle(exc)
        return

    def human(items: list) -> None:
        if not items:
            console.print("no jobs")
            return
        table = Table(box=None, pad_edge=False)
        for col in ("id", "state", "attempts", "machine", "by"):
            table.add_column(col)
        for r in items:
            table.add_row(r["id"], state_text(r["state"]), str(r["attempts"]), r["machine"] or "-", r["submitted_by"])
        console.print(table)

    _emit(context, {"jobs": rows}, lambda p: human(p["jobs"]))


@queue_app.command("cancel")
@_guard
def cancel_command(ctx: typer.Context, job: str = typer.Argument(...)) -> None:
    """Cancel a job: at once when queued, otherwise the coordinator stops it on its machine."""
    context = get_context(ctx)
    try:
        info = api.cancel(job, _project(context))
    except JambuError as exc:
        handle(exc)
        return
    _emit(context, info, _job_human)


@queue_app.command("retry")
@_guard
def retry_command(ctx: typer.Context, job: str = typer.Argument(...)) -> None:
    """Queue a FAILED job again (it keeps its attempts)."""
    context = get_context(ctx)
    try:
        info = api.retry(job, _project(context))
    except JambuError as exc:
        handle(exc)
        return
    _emit(context, info, _job_human)


@queue_app.command("pull")
@_guard
def pull_command(
    ctx: typer.Context,
    job: str = typer.Argument(...),
    to: Optional[Path] = typer.Option(None, "--to", help="Where the outputs land (default gpu-outputs/<job>)."),
) -> None:
    """Get a job's verified outputs."""
    context = get_context(ctx)
    try:
        path = api.outputs(job, to, _project(context))
    except JambuError as exc:
        handle(exc)
        return
    ref = api.status(job, _project(context)).get("outputs_ref") or {}
    files = sorted(f["path"] for f in ref.get("files", []))
    _emit(
        context, {"job_id": job, "path": str(path), "files": files},
        lambda r: console.print(Text(f"+ {len(files)} files in {path}", style="green")),
    )


@queue_app.command("pause")
@_guard
def pause_command(ctx: typer.Context) -> None:
    """Stop starting new jobs (running ones go on)."""
    context = get_context(ctx)
    try:
        result = api.pause(_project(context))
    except JambuError as exc:
        handle(exc)
        return
    _emit(context, result, lambda r: console.print("queue paused"))


@queue_app.command("resume")
@_guard
def resume_command(ctx: typer.Context) -> None:
    """Start jobs again."""
    context = get_context(ctx)
    try:
        result = api.resume(_project(context))
    except JambuError as exc:
        handle(exc)
        return
    _emit(context, result, lambda r: console.print("queue resumed"))


@queue_app.command("drain")
@_guard
def drain_command(
    ctx: typer.Context,
    now: bool = typer.Option(
        False, "--now",
        help="Also send running jobs back to the queue (not failed) and retire every machine, pulling their outputs.",
    ),
) -> None:
    """Pause, and let the machines idle out and stop (with --now: stop them now)."""
    context = get_context(ctx)
    try:
        result = api.drain(_project(context), now=now)
    except JambuError as exc:
        handle(exc)
        return
    _emit(context, result, lambda r: console.print(f"queue paused; drain {r['drain']}"))


@queue_app.command("audit")
@_guard
def audit_command(
    ctx: typer.Context,
    repair: bool = typer.Option(False, "--repair", help="Fix what can be fixed safely."),
) -> None:
    """Check the queue's guarantees (nothing run twice, lost, unverified or orphaned, spend within the cap)."""
    context = get_context(ctx)
    try:
        result = api.audit_queue(_project(context), repair=repair)
    except JambuError as exc:
        handle(exc)
        return

    def human(r: dict) -> None:
        if r["ok"]:
            console.print(Text("+ no violations", style="green"))
        for v in r["violations"]:
            console.print(f"{v['code']}  {v['job_id'] or v['slot'] or ''}  {v['detail']}" + ("  (repaired)" if v["repaired"] else ""))

    _emit(context, result, human)
    if any(not v["repaired"] for v in result["violations"]):
        raise typer.Exit(1)  # something is still wrong


@queue_app.command("stats")
@_guard
def stats_command(ctx: typer.Context) -> None:
    """The coordinator's metrics (as of its last tick) and the queue's counts."""
    context = get_context(ctx)
    try:
        result = api.stats(_project(context))
    except JambuError as exc:
        handle(exc)
        return

    def human(r: dict) -> None:
        for name, value in sorted(r["metrics"].items()):
            console.print(f"{name} {value:g}")
        if not r["metrics"]:
            console.print("no metrics yet: nothing has run `gpu queue run` here")

    _emit(context, result, human)


@queue_app.command("run")
@_guard
def run_command(
    ctx: typer.Context,
    once: bool = typer.Option(False, "--once", help="Run one tick and exit."),
    metrics_port: Optional[int] = typer.Option(
        None, "--metrics-port", help="Serve /metrics on 127.0.0.1:PORT (loopback only).",
    ),
) -> None:
    """Run the coordinator: claim jobs, run them on the pool, deliver outputs. Stops on SIGINT/SIGTERM."""
    from ..queue.coordinator import Coordinator

    context = get_context(ctx)
    server = None
    try:
        rt = api.open_runtime(_project(context))
        pool = api.build_pool(rt.config, rt.observer)
        coordinator = Coordinator(rt.config, rt.backend, rt.outputs, pool, rt.observer)
        if metrics_port is not None:
            if not 0 <= metrics_port <= 65535:
                raise ConfigError("--metrics-port must be a port number")
            server = api.serve_metrics(rt.observer, metrics_port)
            context.echo(f"> metrics on http://127.0.0.1:{server.server_address[1]}/metrics")
        if once:
            try:
                report = coordinator.tick()
                leader = coordinator.leader
            finally:
                coordinator.release()
            result = {**asdict(report), "leader": leader, "worker": coordinator.worker_id}
            _emit(context, result, lambda r: console.print(
                f"tick: {r['dispatched']} dispatched, {r['settled']} settled, {r['requeued']} requeued, {r['opened']} opened"
                + ("" if r["leader"] else " (another coordinator holds the lease)")
            ))
            return
        stop = threading.Event()
        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stop.set())
        context.echo(f"> coordinator {coordinator.worker_id} running; Ctrl-C stops it (machines keep their deadlines)")
        try:
            coordinator.run(stop)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if context.json:
            print_json({"stopped": True, "worker": coordinator.worker_id})
        else:
            context.echo("+ coordinator stopped")
    except JambuError as exc:
        handle(exc)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


@queue_app.command("install-service")
@_guard
def install_service_command(
    ctx: typer.Context,
    uninstall: bool = typer.Option(False, "--uninstall", help="Remove the service files."),
) -> None:
    """Write a user service that keeps `gpu queue run` running in this project (not enabled: the command is printed)."""
    from . import timer

    context = get_context(ctx)
    try:
        project = api.load_queue_config(_project(context)).project_dir
        result = timer.uninstall_queue_service(project) if uninstall else timer.install_queue_service(project)
    except JambuError as exc:
        handle(exc)
        return

    def human(r: dict) -> None:
        if uninstall:
            console.print("removed: " + (", ".join(r["removed"]) or "nothing"))
            console.print(f"then run: {r['disable']}")
        else:
            console.print("wrote: " + ", ".join(r["files"]))
            console.print(f"enable with: {r['enable']}")
            console.print("! credentials must be in ~/.gpu/credentials: a service does not get your shell environment")

    _emit(context, result, human)
