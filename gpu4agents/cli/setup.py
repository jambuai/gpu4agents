"""`gpu setup` / `init` / `offers` - provisioning entry points."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table
from rich.text import Text

from ..core.config import (
    DEFAULT_CONFIG_TEMPLATE,
    DEFAULT_CONFIG_TEMPLATE_FLAT,
    CONFIG_FILENAMES,
)
from ..core.errors import JambuError
from ._common import console, fmt_money, get_context, handle, print_json, register_project


def init_command(
    ctx: typer.Context,
    model: str = typer.Option("empero-ai/Qwythos-9B-v2", "--model", help="HuggingFace model id."),
    provider: str = typer.Option(
        "vast",
        "--provider",
        help="GPU provider. One gpu does not ship is loaded from gpu_providers/NAME.py (gpu providers new NAME).",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config file."),
    flat: bool = typer.Option(
        False,
        "--flat",
        help=(
            "Write a legacy, gpu-runtime-only config.yml (no gpu_runtime: "
            "wrapper) instead of the shared jambu.yaml."
        ),
    ),
) -> None:
    """Write a starter jambu.yaml (and .env.example) in the current directory."""
    from ..core.errors import ConfigError as _ConfigError
    from ..providers import scaffold
    from ..providers.registry import load_builtin_providers

    try:
        credential_names = list(load_builtin_providers().get_class(provider).required_credentials)
        plugin = None
    except _ConfigError:
        # Not built in or installed: an adapter file in this project.
        plugin = scaffold.plugin_spec(provider)
        credential_names = []
    indent = "  " if flat else "    "
    provider_block = f"{indent}name: {provider}\n" + (f"{indent}plugin: {plugin}\n" if plugin else "")
    if flat:
        target = Path.cwd() / "config.yml"
        content = DEFAULT_CONFIG_TEMPLATE_FLAT.format(model_id=model)
    else:
        target = Path.cwd() / CONFIG_FILENAMES[0]  # jambu.yaml
        content = DEFAULT_CONFIG_TEMPLATE.format(model_id=model)
    content = content.replace(f"{indent}name: vast\n", provider_block, 1)

    if target.exists() and not force:
        console.print(Text(f"! {target} already exists (use --force)", style="yellow"))
        raise typer.Exit(1)
    target.write_text(content)
    console.print(Text(f"+ wrote {target}", style="green"))
    if not flat:
        console.print(
            Text(
                "  gpu-runtime config lives under the gpu_runtime: key - other "
                "Jambu Lab tools can share this file",
                style="dim",
            )
        )

    env_example = Path.cwd() / ".env.example"
    if not env_example.exists():
        env_example.write_text(
            "# Credentials never belong in jambu.yaml / config.yml.\n"
            "# `gpu login` keeps the provider key in ~/.gpu/credentials instead.\n"
            + "".join(f"{name}=\n" for name in credential_names)
            + "# Optional, for gated HuggingFace models:\n"
            "HF_TOKEN=\n"
        )
        console.print(Text(f"+ wrote {env_example}", style="green"))
    if plugin and not (Path.cwd() / plugin.split(":")[0]).is_file():
        console.print(
            Text(f"! {provider} is not built in: run `gpu providers new {provider}` and fill in the adapter", style="yellow")
        )
    login = "gpu login" if provider == "vast" else f"gpu login --provider {provider}"
    console.print(f"next: {login} && gpu validate")


def setup_command(
    ctx: typer.Context,
    recreate: bool = typer.Option(
        False, "--recreate", help="Replace an existing instance that no longer matches jambu.yaml."
    ),
    allow_cost_override: bool = typer.Option(
        False, "--allow-cost-override", help="Provision even above budget.max_hourly_cost_usd."
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Return as soon as the instance is created."
    ),
) -> None:
    """Provision the instance, install the watchdog and start the model server."""
    context = get_context(ctx)
    try:
        engine = context.engine()
        register_project(context)
        state = engine.setup(
            recreate=recreate, allow_cost_override=allow_cost_override, wait=not no_wait
        )
    except JambuError as exc:
        handle(exc)
        return

    payload = engine.connection(state)
    watchdog = state.watchdog.to_dict()
    watchdog.pop("token", None)
    payload["watchdog"] = watchdog
    if context.json:
        print_json(payload)
        return
    if state.endpoint:
        console.print(
            f"\n  endpoint  [bold]{state.endpoint}[/bold]\n"
            f"  openai    [bold]{payload['base_url']}[/bold]\n"
            f"  model     [bold]{payload['model']}[/bold]\n"
            "  api key   [dim]gpu env[/dim]"
        )
    console.print("\nnext: gpu run python your_experiment.py")


def offers_command(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", "-n", help="How many offers to show."),
) -> None:
    """Show matching capacity and its price before committing to it."""
    context = get_context(ctx)
    try:
        engine = context.engine()
        engine.provider.require_capability("pricing_query")
        spec = engine.build_spec()
        offers = engine.provider.find_offers(spec, limit=limit)
    except JambuError as exc:
        handle(exc)
        return

    rows = [
        {
            "id": o.id,
            "gpu": o.gpu_name,
            "count": o.gpu_count,
            "vram_gb": o.gpu_vram_gb,
            "disk_gb": o.disk_gb,
            "hourly_cost_usd": o.hourly_cost_usd,
            "region": o.region,
            "cuda_max": o.cuda_max,
        }
        for o in offers
    ]
    if context.json:
        print_json({"offers": rows})
        return

    if not rows:
        console.print(Text("no matching offers", style="yellow"))
        return

    ceiling = engine.config.budget.max_hourly_cost_usd
    table = Table(title=f"{engine.provider.display_name} offers", title_justify="left")
    for column in ("OFFER", "GPU", "N", "VRAM", "DISK", "$/H", "REGION"):
        table.add_column(column)
    for row in rows:
        over = ceiling is not None and row["hourly_cost_usd"] > float(ceiling)
        table.add_row(
            row["id"],
            row["gpu"],
            str(row["count"]),
            f"{row['vram_gb']:.0f} GB",
            f"{row['disk_gb']:.0f} GB",
            Text(
                fmt_money(row["hourly_cost_usd"]) or "-",
                style="red" if over else "green",
            ),
            row["region"] or "-",
        )
    console.print(table)
    if ceiling:
        console.print(Text(f"  budget ceiling: ${float(ceiling):.3f}/h", style="dim"))
