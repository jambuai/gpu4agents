"""`gpu dashboard`: a read-only, host-wide status web UI.

    gpu dashboard start   run the dashboard in the foreground (what the service runs)

The dashboard reads only local state (the project registry + each project's
.gpu/state.json) and probes endpoint reachability; it never calls a provider
API and holds no provider credential. Bound to 127.0.0.1 by default.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import typer

from ..core.durations import parse_duration
from ..core.sweep import ProjectRegistry
from ..dashboard.auth import TokenGate
from ..dashboard.collect import project_snapshots
from ..dashboard.probe import ReachabilityWatch
from ..dashboard.server import DashboardServer
from ._common import err_console, get_context

dashboard_app = typer.Typer(help="Read-only control-plane web UI: every GPU, proxy and service this host sees.")


def _project_paths(context) -> list[Path]:
    paths = list(ProjectRegistry().paths())
    own = getattr(context, "config_path", None)
    if own:
        own = Path(own)
        if own not in paths:
            paths.append(own)
    return paths


def _state_provider(context):
    def provider() -> dict:
        projects = project_snapshots(_project_paths(context), time.time())
        instances = [r for r in projects if r.get("instance_id")]
        return {"instances": instances, "projects": projects, "proxies": []}

    return provider


def _token(cfg) -> str | None:
    return cfg.options.get("token") or os.environ.get("GPU_DASHBOARD_TOKEN")


@dashboard_app.command("start")
def start_command(ctx: typer.Context) -> None:
    """Run the dashboard in the foreground (blocks). This is what the service runs."""
    context = get_context(ctx)
    cfg = context.config().dashboard
    host, port = cfg.listen_host_port
    provider = _state_provider(context)
    gate = TokenGate(_token(cfg))
    watch = ReachabilityWatch(
        interval_s=parse_duration(cfg.watch_interval) or 15.0,
        history_s=parse_duration(cfg.history) or 1200.0,
    )
    watch.start(lambda: [r["endpoint"] for r in provider()["projects"] if r.get("endpoint")])
    if not cfg.enabled:
        context.echo("! dashboard.enabled is false in gpu.yaml; starting anyway because you asked")
    try:
        server = DashboardServer(cfg, provider, watch, gate, log=context.echo)
    except OSError as exc:
        err_console.print(f"error: cannot bind {host}:{port}: {exc}")
        raise typer.Exit(1) from exc
    server.serve_forever()
