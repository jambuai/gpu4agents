"""gpu entrypoint."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Optional

import typer

from ..core.errors import JambuError
from . import account
from . import agent as agent_cli
from . import inspect as inspect_cli
from . import lifecycle, providers, run, setup, status, validate
from ._common import Context, err_console, get_context, print_json

app = typer.Typer(
    name="gpu",
    help=(
        "GPU for Agents: rent a GPU, serve a model, run the work; the machine stops "
        "when nothing uses it.\n\n"
        "jambu.yaml is the desired-state source of truth; .gpu/state.json is what "
        "was last observed."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


def _print_version(value: bool) -> None:
    if value:
        from .. import __version__

        print(__version__)
        raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to jambu.yaml (default: nearest one upwards)."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Machine-readable output on stdout."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose diagnostics."),
    version: bool = typer.Option(
        False, "--version", callback=_print_version, is_eager=True, help="Print the version and exit."
    ),
) -> None:
    context = Context()
    context.config_path = config
    context.json = json_output
    context.verbose = verbose
    ctx.obj = context


app.command("init")(setup.init_command)
app.command("inspect-model")(inspect_cli.inspect_model_command)
app.command("profiles")(inspect_cli.profiles_command)
app.add_typer(providers.providers_app, name="providers")
app.command("validate")(validate.validate_command)
app.command("offers")(setup.offers_command)
app.command("setup")(setup.setup_command)
app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(run.run_command)
app.command("status")(status.status_command)
app.command("logs")(status.logs_command)
app.command("events")(status.events_command)
app.command("lock")(lifecycle.lock_command)
app.command("unlock")(lifecycle.unlock_command)
app.command("stop")(lifecycle.stop_command)
app.command("destroy")(lifecycle.destroy_command)
app.command("guard")(lifecycle.guard_command)
app.command("login")(account.login_command)
app.command("ls")(account.ls_command)
app.command("sweep")(account.sweep_command)
app.command("mcp")(agent_cli.mcp_command)
app.add_typer(agent_cli.agent_app, name="agent")


@app.command("endpoint")
def endpoint_command(ctx: typer.Context) -> None:
    """Print the model endpoint URL (handy for shell pipelines)."""
    context = get_context(ctx)
    try:
        state = context.engine().store.load()
    except JambuError as exc:
        err_console.print(f"error: {exc}")
        raise typer.Exit(getattr(exc, "exit_code", 1)) from exc
    if not state.endpoint:
        err_console.print("error: no endpoint; run `gpu setup` first")
        raise typer.Exit(3)
    print(state.endpoint)


@app.command("env")
def env_command(
    ctx: typer.Context,
    fmt: str = typer.Option(
        "shell", "--format", "-f", help="shell (export lines), dotenv or json."
    ),
) -> None:
    """Print how to reach the model (base URL, API key, model) without starting anything.

    `eval "$(gpu env)"` sets OPENAI_BASE_URL, OPENAI_API_KEY and JAMBU_MODEL_ID,
    which every OpenAI SDK reads. Exit code 3 when there is no endpoint.
    """
    context = get_context(ctx)
    if fmt not in ("shell", "dotenv", "json"):
        err_console.print("error: --format must be shell, dotenv or json")
        raise typer.Exit(2)
    try:
        engine = context.engine()
        conn = engine.connection(engine.store.load())
    except JambuError as exc:
        err_console.print(f"error: {exc}")
        raise typer.Exit(getattr(exc, "exit_code", 1)) from exc
    if context.json or fmt == "json":
        print_json(conn)
        if not conn["endpoint"]:
            raise typer.Exit(3)
        return
    if not conn["endpoint"]:
        err_console.print("error: no endpoint; run `gpu setup` first")
        raise typer.Exit(3)
    values = {
        "OPENAI_BASE_URL": conn["base_url"],
        "OPENAI_API_KEY": conn["api_key"],
        "JAMBU_GPU_ENDPOINT": conn["endpoint"],
        "JAMBU_MODEL_ID": conn["model"],
    }
    prefix = "export " if fmt == "shell" else ""
    for key, value in values.items():
        print(f"{prefix}{key}={shlex.quote(str(value))}")


@app.command("version")
def version_command() -> None:
    """Print the CLI version."""
    from .. import __version__

    print(__version__)


def main() -> None:
    try:
        app()
    except JambuError as exc:  # safety net: errors are normalized, never tracebacks
        err_console.print(f"error: {exc}")
        sys.exit(getattr(exc, "exit_code", 1))
    except KeyboardInterrupt:
        err_console.print("interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
