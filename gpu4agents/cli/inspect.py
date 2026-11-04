"""`gpu inspect-model` - where gpu.yaml's numbers actually come from.

Fetches the two public HuggingFace documents that answer "what should
compute.gpu / runtime look like for this model" (see
gpu4agents/core/model_inspect.py for exactly which fields), and prints a
ready-to-paste gpu_runtime: snippet plus the reasoning behind each value.

`--add-profile` closes the loop: instead of hand-writing a new full gpu.yaml
per model (which only ever grows, never shrinks, and duplicates across every
project directory), it writes the model-specific fragment straight into the
shared gpu.models.yaml catalog under a name - `model_profile: <name>` in a
project's gpu.yaml is then the only thing that ever needs to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.text import Text

from rich.table import Table

from ..core.config import CATALOG_FILENAMES, find_catalog, load_catalog
from ..core.credentials import CredentialResolver
from ..core.errors import ConfigError, JambuError
from ..core.model_inspect import inspect_model
from ._common import console, get_context, handle, print_json


def inspect_model_command(
    ctx: typer.Context,
    repo_id: str = typer.Argument(
        ..., help="HuggingFace repo id to inspect, e.g. Qwen/Qwen2.5-7B-Instruct."
    ),
    revision: str = typer.Option("main", "--revision", help="Repo revision/branch/tag."),
    hf_token_env: str = typer.Option(
        "HF_TOKEN", "--hf-token-env", help="Env var holding a token for gated repos."
    ),
    add_profile: str = typer.Option(
        "",
        "--add-profile",
        help=(
            "Write the result straight into gpu.models.yaml under this name "
            "instead of just printing a snippet (creates the catalog if none "
            "exists yet). Use model_profile: <name> in gpu.yaml to consume it."
        ),
    ),
    catalog_path_opt: Optional[Path] = typer.Option(
        None,
        "--catalog",
        help=f"Catalog file to write to with --add-profile (default: nearest {CATALOG_FILENAMES[0]}, or a new one here).",
    ),
    force: bool = typer.Option(
        False, "--force", help="With --add-profile, overwrite an existing profile of that name."
    ),
) -> None:
    """Inspect a HuggingFace repo and suggest gpu.yaml values for it.

    Reads exactly two public documents, cited in the output:

    \b
      https://huggingface.co/api/models/<repo>            (param count, dtype)
      https://huggingface.co/<repo>/raw/<rev>/config.json  (architecture, context)

    This is a starting point, not an oracle - the VRAM figure is weights plus
    a flat margin for KV cache/activations (which depend on context length
    and concurrency this command has no way to know). Confirm with
    `gpu offers` and a real health check before trusting it in production.
    """
    context = get_context(ctx)
    resolver = CredentialResolver(Path.cwd())
    token = resolver.get(hf_token_env)

    try:
        result = inspect_model(repo_id, revision=revision, token=token)
    except JambuError as exc:
        handle(exc)
        return

    if add_profile:
        try:
            target = _write_profile(result, add_profile, catalog_path_opt, force=force)
        except JambuError as exc:
            handle(exc)
            return
        if context.json:
            print_json({**result.to_dict(), "profile_name": add_profile, "catalog_path": str(target)})
        else:
            console.print(Text(f"+ wrote profile '{add_profile}' to {target}", style="green"))
            console.print(f"  use it with: model_profile: {add_profile}")
        return

    if context.json:
        print_json(result.to_dict())
        return

    console.print(f"[bold]{result.repo_id}[/bold] @ {result.revision}")
    console.print(
        f"  architecture   {', '.join(result.architectures) or '?'} "
        f"(model_type: {result.model_type or '?'})"
    )
    if result.total_params:
        console.print(
            f"  parameters     {result.total_params / 1e9:.2f}B "
            f"({result.param_dtype or '?'}"
            + (f", quantization: {result.quant_method}" if result.quant_method else "")
            + ")"
        )
        console.print(
            f"  weight size    ~{result.weight_gb:.1f} GB "
            f"-> suggested min_vram_gb: {result.suggested_min_vram_gb}"
        )
    else:
        console.print(Text("  parameters     unknown (no safetensors metadata)", style="yellow"))
    console.print(f"  context        native {result.max_position_embeddings or '?'}")
    console.print(
        f"  tool calling   {result.suggested_tool_call_parser or '(no parser guessed)'}"
    )
    if result.gated:
        console.print(Text("  gated          yes - a real HF token is required", style="yellow"))
    if result.is_hybrid_attention:
        console.print(
            Text("  hybrid attention  yes - needs Ampere-or-newer GPU", style="yellow")
        )

    for note in result.notes:
        console.print(Text(f"  ! {note}", style="yellow"))

    console.print("\n[bold]Sources[/bold]")
    console.print(f"  https://huggingface.co/api/models/{repo_id}")
    console.print(f"  https://huggingface.co/{repo_id}/raw/{revision}/config.json")

    console.print("\n[bold]Suggested gpu_runtime: snippet[/bold]")
    console.print(result.to_yaml_snippet())
    console.print(
        Text(
            f"\nTip: `gpu inspect-model {repo_id} --add-profile <name>` writes this "
            "straight into your model catalog instead of a one-off file.",
            style="dim",
        )
    )


def profiles_command(
    ctx: typer.Context,
    catalog_path_opt: Optional[Path] = typer.Option(
        None, "--catalog", help=f"Catalog to list (default: nearest {CATALOG_FILENAMES[0]})."
    ),
) -> None:
    """List the model profiles available in the model catalog (gpu.models.yaml)."""
    context = get_context(ctx)
    target = catalog_path_opt or find_catalog(Path.cwd())
    if target is None or not target.is_file():
        if context.json:
            print_json({"catalog": None, "profiles": {}})
        else:
            console.print(
                Text(
                    f"no {CATALOG_FILENAMES[0]} found in this directory or any parent. "
                    "Create one with `gpu inspect-model <repo> --add-profile <name>`.",
                    style="yellow",
                )
            )
        return

    try:
        catalog = load_catalog(target)
    except JambuError as exc:
        handle(exc)
        return

    if context.json:
        print_json({"catalog": str(target), "profiles": catalog})
        return

    table = Table(title=f"Model profiles ({target})", title_justify="left")
    for column in ("PROFILE", "MODEL", "MIN VRAM", "TOOL CALLING"):
        table.add_column(column)
    for name, fragment in sorted(catalog.items()):
        if not isinstance(fragment, dict):
            continue
        model_id = (fragment.get("model") or {}).get("id", "?")
        vram = ((fragment.get("compute") or {}).get("gpu") or {}).get("min_vram_gb", "-")
        runtime = fragment.get("runtime") or {}
        tool_calling = runtime.get("tool_call_parser") or runtime.get("tool_calling", "-")
        table.add_row(name, str(model_id), str(vram), str(tool_calling))
    console.print(table)
    console.print(Text(f"  use with: model_profile: <name> in gpu.yaml", style="dim"))


def _write_profile(result, name: str, catalog_path_opt: Optional[Path], *, force: bool) -> Path:
    if catalog_path_opt:
        target = catalog_path_opt
    else:
        target = find_catalog(Path.cwd()) or (Path.cwd() / CATALOG_FILENAMES[0])

    catalog: dict = load_catalog(target) if target.is_file() else {}
    if name in catalog and not force:
        raise ConfigError(
            f"profile '{name}' already exists in {target} (use --force to overwrite)"
        )
    catalog[name] = result.to_profile_fragment()
    target.write_text(yaml.safe_dump(catalog, sort_keys=False))
    return target
