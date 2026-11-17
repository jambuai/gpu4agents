"""`gpu claude`: launch Claude Code wired to this project's model.

Turnkey: `gpu setup` serves the model behind the gateway; `gpu claude` reads
the same gpu.yaml, makes sure the gateway is up, and execs `claude` with the
Anthropic env it needs (base URL -> gateway, model + aliases, context window).
The gpu.yaml is the single source of truth; there is no separate settings file.

The gateway (adapter ``anthropic-openai``) is what lets Claude Code's Anthropic
wire talk to an OpenAI backend such as the ``llama`` engine; against a backend
that already speaks ``/v1/messages`` natively (vLLM) the ``anthropic`` adapter
works too.
"""

from __future__ import annotations

import http.client
import os
import shutil
import subprocess
import sys
import time
from typing import List

import typer

from ..core.config import RuntimeConfig
from ..core.errors import JambuError
from ._common import console, err_console, get_context


def claude_env(config: RuntimeConfig) -> dict[str, str]:
    """The Anthropic env Claude Code needs to use this project's model, derived
    entirely from gpu.yaml (gateway listen address, model id, context length)."""
    host, port = config.gateway.listen_host_port
    base_url = f"http://{host}:{port}"
    model = config.model.served_name or config.model.id
    ctx = config.runtime.context_length or 131072
    max_out = max(4096, min(32000, ctx // 8))
    return {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        # Claude Code assumes a 1M window for an unknown model id; pin it to what
        # the server actually serves so it manages context correctly.
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(ctx),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_out),
        # Skip Claude Code's non-essential model calls (title generation, etc.)
        # that a small local model serves slowly.
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
    }


def _listening(host: str, port: int) -> bool:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=2)
        conn.request("HEAD", "/")
        ok = conn.getresponse().status == 200
        conn.close()
        return ok
    except OSError:
        return False


def _ensure_gateway(context, host: str, port: int) -> None:
    """Start the gateway in the background if it is not already listening."""
    if _listening(host, port):
        return
    gpu_bin = shutil.which("gpu") or sys.argv[0]
    project_dir = str(context.config().project_dir)
    console.print(f"[dim]starting gateway on {host}:{port} ...[/dim]")
    subprocess.Popen(
        [gpu_bin, "gateway", "start"],
        cwd=project_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # survives this process exec'ing into claude
    )
    for _ in range(20):
        if _listening(host, port):
            return
        time.sleep(0.5)
    err_console.print(f"! gateway did not come up on {host}:{port}; run `gpu gateway start` manually")


def claude_command(
    ctx: typer.Context,
    args: List[str] = typer.Argument(None, help="Arguments passed through to `claude`."),
) -> None:
    """Ensure the gateway is up, then launch Claude Code wired to this model."""
    context = get_context(ctx)
    try:
        config = context.config()
    except JambuError as exc:
        err_console.print(f"error: {exc}")
        raise typer.Exit(2) from exc

    if config.gateway.adapter not in ("anthropic-openai", "anthropic"):
        context.echo(
            f"! gateway.adapter is '{config.gateway.adapter}'; Claude Code needs "
            "'anthropic-openai' (OpenAI backend) or 'anthropic' (native)"
        )
    host, port = config.gateway.listen_host_port
    _ensure_gateway(context, host, port)

    claude_bin = shutil.which("claude")
    if not claude_bin:
        err_console.print("error: `claude` (Claude Code) is not on PATH")
        raise typer.Exit(127)

    env = {**os.environ, **claude_env(config)}
    argv = [claude_bin, *(args or [])]
    os.execve(claude_bin, argv, env)  # replace this process with Claude Code
