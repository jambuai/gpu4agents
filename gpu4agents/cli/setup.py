"""`gpu setup` / `init` / `offers` - provisioning entry points."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table
from rich.text import Text

import shutil
from typing import Optional

from ..core.config import (
    CUSTOM_CONFIG_TEMPLATE,
    DEFAULT_CONFIG_TEMPLATE,
    DEFAULT_CONFIG_TEMPLATE_FLAT,
    CONFIG_FILENAMES,
    LEGACY_CONFIG_FILENAMES,
    JOBS_CONFIG_TEMPLATE,
    MODEL_TASKS,
    SERVE_CONFIG_TEMPLATE,
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
    jobs: bool = typer.Option(
        False,
        "--jobs",
        help="A machine that serves no model and only runs `gpu job` (runtime.engine: none).",
    ),
    engine: str = typer.Option(
        "auto",
        "--engine",
        help=(
            "auto (from the model card: chat models on vllm, any other task on hf), "
            "vllm, hf, custom (a server you write, SERVING.md) or none (same as --jobs)."
        ),
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="With hf: chat, embeddings, image, transcription, speech or pipeline (default: from the model card).",
    ),
    service_dir: str = typer.Option(
        "server", "--service-dir", help="With --engine custom: where your server lives."
    ),
    min_vram_gb: Optional[int] = typer.Option(
        None, "--min-vram-gb", help="Minimum GPU memory to rent (default 24, or sized from the model)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config file."),
    flat: bool = typer.Option(
        False,
        "--flat",
        help=(
            "Write a legacy, gpu-runtime-only config.yml (no gpu_runtime: "
            "wrapper) instead of the shared gpu.yaml."
        ),
    ),
) -> None:
    """Write a starter gpu.yaml (and .env.example) in the current directory.

    Any Hugging Face model works: a chat model is served by vLLM; an image,
    speech, transcription, embeddings or other pipeline model by gpu's own
    OpenAI-compatible server (runtime.engine: hf). The model card decides,
    unless --engine says otherwise.
    """
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
    if engine not in ("auto", "vllm", "hf", "custom", "none"):
        console.print(Text("! --engine must be auto, vllm, hf, custom or none", style="yellow"))
        raise typer.Exit(2)
    if task is not None and task not in MODEL_TASKS:
        console.print(Text(f"! --task must be one of {', '.join(MODEL_TASKS)}", style="yellow"))
        raise typer.Exit(2)
    if jobs:
        engine = "none"
    if flat and engine not in ("auto", "vllm"):
        console.print(Text("! only a vllm config can be --flat; drop --flat", style="yellow"))
        raise typer.Exit(2)
    model_given = ctx.get_parameter_source("model") is not None and (
        ctx.get_parameter_source("model").name != "DEFAULT"
    )
    sized: Optional[int] = None
    if engine == "auto" or (engine == "hf" and task is None):
        detected_engine, detected_task, sized, note = detect_serving(model, model_given)
        if engine == "auto":
            engine = detected_engine
        task = task or detected_task
        if note:
            console.print(Text(f"  {note}", style="dim"))
    vram = min_vram_gb or sized or 24
    if flat:
        target = Path.cwd() / "config.yml"
        content = DEFAULT_CONFIG_TEMPLATE_FLAT.format(model_id=model)
    elif engine == "none":
        target = Path.cwd() / CONFIG_FILENAMES[0]
        content = JOBS_CONFIG_TEMPLATE.format(min_vram_gb=vram)
    elif engine == "hf":
        target = Path.cwd() / CONFIG_FILENAMES[0]
        task = task or "auto"
        content = SERVE_CONFIG_TEMPLATE.format(
            model_id=model,
            min_vram_gb=vram,
            task=task,
            task_note=(
                "From the model card. chat, embeddings, image, transcription, speech or pipeline."
                if task != "auto"
                else "auto: read from the model card when the server starts."
            ),
        )
    elif engine == "custom":
        target = Path.cwd() / CONFIG_FILENAMES[0]
        content = CUSTOM_CONFIG_TEMPLATE.format(
            model_id=model if model_given else '""', min_vram_gb=vram, service_dir=service_dir
        )
    else:
        target = Path.cwd() / CONFIG_FILENAMES[0]  # gpu.yaml
        content = DEFAULT_CONFIG_TEMPLATE.format(model_id=model)
    content = content.replace("min_vram_gb: 24\n", f"min_vram_gb: {vram}\n", 1)
    content = content.replace(f"{indent}name: vast\n", provider_block, 1)

    if target.exists() and not force:
        console.print(Text(f"! {target} already exists (use --force)", style="yellow"))
        raise typer.Exit(1)
    legacy = [Path.cwd() / name for name in LEGACY_CONFIG_FILENAMES if (Path.cwd() / name).is_file()]
    if legacy and not flat:
        if not force:
            console.print(
                Text(
                    f"! {legacy[0].name} already exists here: rename it to gpu.yaml, "
                    "or use --force to write a new gpu.yaml that takes its place",
                    style="yellow",
                )
            )
            raise typer.Exit(1)
        console.print(Text(f"  {legacy[0].name} is left as it was; gpu.yaml is read first", style="dim"))
    target.write_text(content)
    console.print(Text(f"+ wrote {target}", style="green"))
    if engine == "custom":
        scaffold_server(Path.cwd() / service_dir)
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
            "# Credentials never belong in gpu.yaml / config.yml.\n"
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
    if engine == "none":
        console.print("then: gpu job run -- python your_script.py   (see `gpu job --help`)")
    elif engine == "hf":
        console.print(f"then: gpu setup   (serves {model} as {task}; routes in `gpu env --format json`)")
    elif engine == "custom":
        console.print(f"then: write your model into {service_dir}/server.py, and gpu setup")


def detect_serving(model: str, model_given: bool) -> tuple[str, Optional[str], Optional[int], str]:
    """(engine, task, min VRAM GB or None, a note) from the model's Hugging Face card."""
    from ..core.model_inspect import ModelInspectError, fetch_hf_api_metadata
    from ..serve import serving_for

    import os

    if not model_given or os.environ.get("GPU_NO_MODEL_CARD"):
        # The default model is a chat model; GPU_NO_MODEL_CARD=1 works offline.
        return "vllm", "chat", None, ""

    try:
        card = fetch_hf_api_metadata(model, token=os.environ.get("HF_TOKEN") or None, timeout=8.0)
    except ModelInspectError as exc:
        return "vllm", None, None, f"could not read the model card ({exc}); assuming a chat model on vLLM"
    engine, task = serving_for(card)
    sized = None
    total = (card.get("safetensors") or {}).get("total")
    if total:
        # bf16 weights plus headroom; the same floor `gpu inspect-model` uses.
        sized = max(8, int(total * 2 / 1e9 * 1.3 + 2.999))
    tag = card.get("pipeline_tag") or card.get("library_name") or "no pipeline_tag"
    if engine == "vllm":
        return engine, task, None, f"{model}: {tag} -> vLLM"
    note = f"{model}: {tag} -> runtime.engine: hf, model.task: {task}"
    library = card.get("library_name")
    if library not in ("transformers", "diffusers", "sentence-transformers"):
        note += (
            f". Its library is {library or 'not declared'}: if the hf server cannot load it, "
            "serve it with your own server (gpu init --engine custom, SERVING.md)"
        )
    return engine, task, sized, note


def scaffold_server(directory: Path) -> None:
    """A starting server for runtime.engine: custom (serve/template/)."""
    from ..serve import TEMPLATE_DIR

    if (directory / "gpujob.yaml").exists() or (directory / "server.py").exists():
        console.print(Text(f"= {directory} already has a server; left as is", style="dim"))
        return
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("server.py", "gpujob.yaml"):
        shutil.copyfile(TEMPLATE_DIR / name, directory / name)
    console.print(Text(f"+ wrote {directory}/server.py and gpujob.yaml (contract: SERVING.md)", style="green"))


def setup_command(
    ctx: typer.Context,
    recreate: bool = typer.Option(
        False, "--recreate", help="Replace an existing instance that no longer matches gpu.yaml."
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

    _print_connection(context, engine, state)


def _print_connection(context, engine, state) -> None:
    payload = engine.connection(state)
    watchdog = state.watchdog.to_dict()
    watchdog.pop("token", None)
    payload["watchdog"] = watchdog
    if context.json:
        print_json(payload)
        return
    if state.endpoint:
        routes = ", ".join(payload.get("routes") or []) or "see GET /v1/models"
        console.print(
            f"\n  endpoint  [bold]{state.endpoint}[/bold]\n"
            f"  openai    [bold]{payload['base_url']}[/bold]\n"
            f"  model     [bold]{payload['model']}[/bold]\n"
            f"  routes    {routes}\n"
            "  api key   [dim]gpu env[/dim]"
        )
    console.print("\nnext: gpu run python your_experiment.py")


def redeploy_command(ctx: typer.Context) -> None:
    """Restart the model server on the running machine (runtime.engine: hf or custom).

    `gpu setup` already redeploys when the model, its settings or your server's
    code changed; this restarts it even when nothing did.
    """
    context = get_context(ctx)
    try:
        engine = context.engine()
        state = engine.redeploy()
    except JambuError as exc:
        handle(exc)
        return
    _print_connection(context, engine, state)


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
