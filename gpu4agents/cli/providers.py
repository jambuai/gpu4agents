"""`gpu providers` - capability matrix (spec section 7), and `gpu providers new`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich.text import Text

from ..core.credentials import CredentialResolver
from ..core.errors import ConfigError, JambuError
from ..providers import scaffold
from ..providers.registry import load_builtin_providers
from ._common import get_context, handle, print_json

providers_app = typer.Typer(
    help="List GPU providers, or start an adapter for a new one.",
    invoke_without_command=True,
)


@providers_app.callback()
def providers_callback(ctx: typer.Context) -> None:
    """List available providers and what each one can do.

    Vast.ai is built in. Any other provider plugs in through the provider
    contract: `gpu providers new NAME` writes the adapter to fill in.
    """
    if ctx.invoked_subcommand is None:
        providers_command(ctx)


@providers_app.command("new")
def new_command(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Short provider name, e.g. runpod or lambda."),
    display_name: str = typer.Option("", "--display-name", help="Human name, e.g. RunPod."),
    directory: Optional[Path] = typer.Option(
        None, "--dir", help="Project directory (default: the jambu.yaml's, else the current one)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing adapter file."),
) -> None:
    """Write gpu_providers/NAME.py, an adapter skeleton for a provider gpu does not ship.

    Fill in its methods from the provider's API docs, then select it in
    jambu.yaml with provider.name and provider.plugin.
    """
    context = get_context(ctx)
    try:
        if directory is None:
            try:
                directory = context.config().project_dir
            except ConfigError:
                directory = Path.cwd()
        path = scaffold.write_template(name, directory, display_name, force)
    except JambuError as exc:
        handle(exc)
        return
    result = {
        "path": str(path),
        "provider": {"name": name, "plugin": scaffold.plugin_spec(name)},
        "next": [
            f"implement every NotImplementedError in {path}",
            f"jambu.yaml: gpu_runtime.provider.name: {name} and plugin: {scaffold.plugin_spec(name)}",
            f"gpu login --provider {name}",
            "gpu validate, then gpu offers (rents nothing)",
        ],
    }
    if context.json:
        print_json(result)
        return
    from ._common import console

    console.print(Text(f"+ wrote {path}", style="green"))
    console.print("  next:")
    for step in result["next"]:
        console.print(f"  - {step}")


def providers_command(ctx: typer.Context) -> None:
    """List available providers and what each one can do."""
    context = get_context(ctx)
    registry = load_builtin_providers()

    try:
        config = context.config()
        project_dir = config.project_dir
        profile = config.provider.profile
        selected = config.provider.name
    except ConfigError:
        config = None
        project_dir = None
        profile = "default"
        selected = None
    if config is not None:
        try:
            registry.class_for(config)  # a plugin adapter joins the list
        except ConfigError as exc:
            handle(exc)

    resolver = CredentialResolver(project_dir, profile=profile)
    rows = []
    for provider_cls in registry.classes():
        missing = [n for n in provider_cls.required_credentials if not resolver.get(n)]
        rows.append(
            {
                "provider": provider_cls.name,
                "display_name": provider_cls.display_name,
                "selected": provider_cls.name == selected,
                "status": "configured" if not missing else "unavailable",
                "missing_credentials": missing,
                "required_credentials": list(provider_cls.required_credentials),
                "capabilities": dict(provider_cls.capabilities.__dict__),
            }
        )

    if context.json:
        print_json({"providers": rows})
        return

    table = Table(title="Providers", title_justify="left")
    table.add_column("PROVIDER")
    table.add_column("STATUS")
    table.add_column("GPU")
    table.add_column("STOP")
    table.add_column("DESTROY")
    table.add_column("SSH")
    table.add_column("PORTS")
    table.add_column("SPOT")
    table.add_column("PRICING")
    for row in rows:
        caps = row["capabilities"]
        name = Text(row["provider"] + (" *" if row["selected"] else ""))
        status = Text(
            row["status"],
            style="green" if row["status"] == "configured" else "yellow",
        )
        table.add_row(
            name,
            status,
            _yn(caps["gpu_selection"]),
            _yn(caps["stop"]),
            _yn(caps["destroy"]),
            _yn(caps["ssh"]),
            _yn(caps["public_ports"]),
            _yn(caps["spot_instances"]),
            _yn(caps["pricing_query"]),
        )
    from ._common import console

    console.print(table)
    for row in rows:
        if row["missing_credentials"]:
            console.print(
                Text(
                    f"  {row['provider']}: set "
                    + ", ".join(row["missing_credentials"]),
                    style="yellow",
                )
            )
    if selected:
        console.print(Text(f"  * selected in jambu.yaml ({selected})", style="dim"))
    console.print(Text("  another provider: gpu providers new NAME", style="dim"))


def _yn(value: bool) -> str:
    return "yes" if value else "no"
