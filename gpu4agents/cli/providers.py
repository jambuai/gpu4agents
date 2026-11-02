"""`gpu providers` - capability matrix (spec section 7)."""

from __future__ import annotations

import typer
from rich.table import Table
from rich.text import Text

from ..core.credentials import CredentialResolver
from ..core.errors import ConfigError
from ..providers.registry import load_builtin_providers
from ._common import get_context, print_json


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


def _yn(value: bool) -> str:
    return "yes" if value else "no"
