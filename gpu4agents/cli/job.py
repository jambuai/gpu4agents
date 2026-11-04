"""`gpu job` - run any program on the GPU machine and bring its outputs back."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from rich.text import Text

from ..core.errors import ConfigError, JambuError
from ._common import console, fmt_money, get_context, handle, print_json, register_project

job_app = typer.Typer(
    help=(
        "Run any program on the rented GPU: upload a directory of code, run a command "
        "there, follow its log, download what it wrote to $JAMBU_OUTPUT_DIR. The contract "
        "is in JOBS.md."
    ),
    no_args_is_help=True,
)

# Exit codes for jobs that did not end on their own.
EXIT_TIMEOUT = 124
EXIT_CANCELLED = 130
EXIT_LOST = 1


def _parse_env(pairs: Optional[List[str]]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ConfigError(f"--env takes KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


def shell_command(argv: Optional[List[str]]) -> Optional[str]:
    """One argument is a shell script (`"pip install x && python a.py"`);
    several are an argv, quoted so the shell sees them unchanged."""
    if not argv:
        return None
    return argv[0] if len(argv) == 1 else shlex.join(argv)


def _exit_code(info: dict) -> int:
    state = info.get("state")
    if state == "succeeded":
        return 0
    if state == "timeout":
        return EXIT_TIMEOUT
    if state == "cancelled":
        return EXIT_CANCELLED
    code = info.get("returncode")
    return int(code) if code else EXIT_LOST


def _summary(info: dict) -> str:
    parts = [f"job {info.get('id')} {info.get('state')}"]
    if info.get("returncode") is not None:
        parts.append(f"exit {info['returncode']}")
    if info.get("elapsed_s") is not None:
        parts.append(f"{info['elapsed_s']:.0f}s")
    if info.get("cost_usd") is not None:
        parts.append(f"~{fmt_money(info['cost_usd'])}")
    return ", ".join(parts)


@job_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_command(
    ctx: typer.Context,
    command: Optional[List[str]] = typer.Argument(
        None,
        help="Shell command to run on the GPU, e.g. `python train.py`. Put `--` before it. "
        "Default: run: in gpujob.yaml.",
    ),
    directory: Path = typer.Option(Path("."), "--dir", "-d", help="Directory of code to upload."),
    setup: Optional[str] = typer.Option(
        None, "--setup", help="Shell command run first, e.g. `pip install -r requirements.txt`."
    ),
    env: Optional[List[str]] = typer.Option(None, "--env", "-e", help="KEY=VALUE for the job (repeatable)."),
    timeout: Optional[str] = typer.Option(None, "--timeout", help="Stop the job after this long, e.g. 2h."),
    label: Optional[str] = typer.Option(None, "--label", help="Name shown in status and logs."),
    exclude: Optional[List[str]] = typer.Option(
        None, "--exclude", help="Path or glob to leave out of the upload (repeatable)."
    ),
    detach: bool = typer.Option(
        False, "--detach", help="Return once the job starts; follow it with `gpu job logs -f`."
    ),
    no_start: bool = typer.Option(
        False,
        "--no-start",
        help="Fail (exit 3) instead of renting or restarting a machine. For callers that "
        "start it with `gpu setup` first.",
    ),
    pull: bool = typer.Option(True, "--pull/--no-pull", help="Download the outputs when the job ends."),
    out: Optional[Path] = typer.Option(None, "--out", help="Where outputs land (default gpu-outputs/<job id>)."),
) -> None:
    """Upload a directory, run a command on the GPU, follow it, bring the outputs back.

    Exit code: the job's own. 124 when it hit --timeout, 130 when cancelled.
    """
    context = get_context(ctx)
    try:
        engine = context.engine()
        if not no_start:
            register_project(context)  # a job provisions when nothing is up
        record = engine.job_submit(
            directory,
            shell_command(command),
            setup=setup,
            env=_parse_env(env),
            timeout=timeout,
            label=label,
            exclude=exclude or [],
            start=not no_start,
        )
    except JambuError as exc:
        handle(exc)
        return
    job_id = record["id"]
    if detach:
        if context.json:
            print_json({"job_id": job_id, "state": record.get("state"), "instance_id": record.get("instance_id")})
        else:
            console.print(f"follow: gpu job logs {job_id} -f   outputs: gpu job pull {job_id}")
        return

    def write(text: str) -> None:
        if not context.json:
            sys.stdout.write(text)
            sys.stdout.flush()

    try:
        info = engine.job_follow(job_id, write)
    except KeyboardInterrupt:
        console.print(
            Text(
                f"\n! stopped following; job {job_id} keeps running on the machine. "
                f"`gpu job logs {job_id} -f` to follow, `gpu job cancel {job_id}` to stop it",
                style="yellow",
            )
        )
        raise typer.Exit(EXIT_CANCELLED)
    except JambuError as exc:
        handle(exc)
        return

    result: dict = {"job": info}
    if pull:
        try:
            result["pulled"] = engine.job_pull(job_id, out)
        except JambuError as exc:
            result["pull_error"] = str(exc)
    if context.json:
        print_json(result)
    else:
        console.print(Text(_summary(info), style="green" if info.get("state") == "succeeded" else "red"))
        pulled = result.get("pulled")
        if pulled:
            console.print(f"  outputs: {len(pulled['files'])} files in {pulled['outputs_dir']}")
        elif result.get("pull_error"):
            console.print(Text(f"  ! outputs not downloaded: {result['pull_error']}", style="yellow"))
    code = _exit_code(info)
    if code:
        raise typer.Exit(code)


@job_app.command("status")
def status_command(
    ctx: typer.Context,
    job_id: Optional[str] = typer.Argument(None, help="Job id (default: the latest)."),
) -> None:
    """State, exit code, run time, cost so far and output files of a job."""
    context = get_context(ctx)
    try:
        info = context.engine().job_info(job_id)
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json(info)
        return
    console.print(_summary(info))
    console.print(f"  command: {info.get('command')}")
    for item in info.get("outputs") or []:
        console.print(f"  out/{item['path']}  ({item['bytes']} bytes)")


@job_app.command("logs")
def logs_command(
    ctx: typer.Context,
    job_id: Optional[str] = typer.Argument(None, help="Job id (default: the latest)."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Keep printing until the job ends."),
) -> None:
    """The job's output (stdout and stderr together)."""
    context = get_context(ctx)
    try:
        engine = context.engine()
        if follow and not context.json:
            resolved = engine.job_log(job_id, 0, 1)["job_id"]
            info = engine.job_follow(resolved, lambda text: (sys.stdout.write(text), sys.stdout.flush()))
            console.print(Text(_summary(info), style="dim"))
            return
        data = engine.job_log(job_id)
        while data.get("offset", 0) < data.get("size", 0):
            more = engine.job_log(data["job_id"], data["offset"])
            if more["offset"] == data["offset"]:
                break
            more["data"] = data["data"] + more["data"]
            data = more
    except KeyboardInterrupt:
        raise typer.Exit(EXIT_CANCELLED)
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json(data)
    else:
        sys.stdout.write(data.get("data") or "")


@job_app.command("pull")
def pull_command(
    ctx: typer.Context,
    job_id: Optional[str] = typer.Argument(None, help="Job id (default: the latest)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Destination (default gpu-outputs/<job id>)."),
) -> None:
    """Download what the job wrote to $JAMBU_OUTPUT_DIR (works while it still runs)."""
    context = get_context(ctx)
    try:
        result = context.engine().job_pull(job_id, out)
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json(result)
    else:
        console.print(f"+ {len(result['files'])} files in {result['outputs_dir']}")


@job_app.command("cancel")
def cancel_command(
    ctx: typer.Context,
    job_id: Optional[str] = typer.Argument(None, help="Job id (default: the latest)."),
) -> None:
    """Stop a running job (SIGTERM, then SIGKILL after 10 s). Its outputs stay."""
    context = get_context(ctx)
    try:
        result = context.engine().job_cancel(job_id)
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json(result)
    else:
        console.print(f"- job {result['job_id']}: {result.get('state')}")


@job_app.command("ls")
def ls_command(ctx: typer.Context) -> None:
    """Jobs of this project, newest last."""
    context = get_context(ctx)
    try:
        records = context.engine().job_list()
    except JambuError as exc:
        handle(exc)
        return
    if context.json:
        print_json({"jobs": records})
        return
    if not records:
        console.print("no jobs yet: gpu job run -- python your_script.py")
        return
    table = Table(box=None, pad_edge=False)
    for column in ("id", "state", "exit", "time", "cost", "label / command"):
        table.add_column(column)
    for r in records:
        elapsed = r.get("elapsed_s")
        table.add_row(
            str(r.get("id")),
            str(r.get("state") or "?"),
            "" if r.get("returncode") is None else str(r["returncode"]),
            "" if elapsed is None else f"{elapsed:.0f}s",
            fmt_money(r.get("cost_usd")) or "",
            str(r.get("label") or r.get("command") or ""),
        )
    console.print(table)
