"""`gpu dashboard`: a read-only, host-wide status web UI.

    gpu dashboard start       run the dashboard in the foreground (what the service runs)
    gpu dashboard install     install + enable a systemd user service (auto-restart, boot-safe)
    gpu dashboard uninstall   remove the service
    gpu dashboard status       is it installed / listening
    gpu dashboard logs         journalctl for the service

The dashboard reads only local state (the project registry + each project's
.gpu/state.json) and probes endpoint reachability; it never calls a provider
API and holds no provider credential. Bound to 127.0.0.1 by default; a
non-loopback bind requires a token (generated on install).
"""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Optional

import typer

from ..core.durations import parse_duration
from ..core.errors import JambuError
from ..core.sweep import ProjectRegistry
from ..dashboard.auth import TokenGate
from ..dashboard.collect import project_snapshots, select_instances
from ..dashboard.probe import ReachabilityWatch
from ..dashboard.server import DashboardServer
from ..dashboard import service as svc
from ._common import console, err_console, get_context, print_json

dashboard_app = typer.Typer(help="Read-only control-plane web UI: every GPU, proxy and service this host sees.")

_LOOPBACK = {"127.0.0.1", "localhost", "::1", ""}


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
        return {"instances": select_instances(projects), "projects": projects, "proxies": []}

    return provider


def _token_file(project_dir: Path) -> Path:
    return Path(project_dir) / ".gpu" / "dashboard.token"


def _read_token(cfg, project_dir: Optional[Path]) -> Optional[str]:
    token = cfg.options.get("token") or os.environ.get("GPU_DASHBOARD_TOKEN")
    if token:
        return token
    if project_dir:
        tf = _token_file(project_dir)
        if tf.is_file():
            return tf.read_text().strip() or None
    return None


def _build_actions(context) -> dict:
    """Phase-3 control actions, wired to THIS project's engine only.

    Mutating another project would need that project's own engine and
    credentials in the service environment; that larger surface is deferred, so
    control acts on the project the dashboard runs in. Enabling control means
    the service process can stop/destroy a GPU and therefore needs provider
    credentials — a deliberate choice, off by default.
    """
    def stop(_project):
        return context.engine().stop(reason="dashboard")

    def destroy(_project):
        return context.engine().destroy(reason="dashboard")

    def lock(_project):
        engine = context.engine()
        return engine.lock(engine.config.lock_default_ttl_s, reason="dashboard")

    def restart_proxy(_project):
        import subprocess

        from ..gateway import service as gw_svc

        unit = gw_svc.unit_name(context.config().project_dir) + ".service"
        subprocess.run(["systemctl", "--user", "restart", unit], check=True)
        return {"restarted": unit}

    return {"stop": stop, "destroy": destroy, "lock": lock, "restart-proxy": restart_proxy}


@dashboard_app.command("start")
def start_command(ctx: typer.Context) -> None:
    """Run the dashboard in the foreground (blocks). This is what the service runs."""
    context = get_context(ctx)
    cfg = context.config().dashboard
    host, port = cfg.listen_host_port
    provider = _state_provider(context)
    gate = TokenGate(_read_token(cfg, context.config().project_dir))
    watch = ReachabilityWatch(
        interval_s=parse_duration(cfg.watch_interval) or 15.0,
        history_s=parse_duration(cfg.history) or 1200.0,
    )
    watch.start(lambda: [r["endpoint"] for r in provider()["projects"] if r.get("endpoint")])
    actions = _build_actions(context) if cfg.control_enabled else None
    if cfg.control_enabled:
        context.echo("! dashboard.control_enabled is true: mutating routes are live (token + confirm required)")
    if not cfg.enabled:
        context.echo("! dashboard.enabled is false in gpu.yaml; starting anyway because you asked")
    local_project = Path(context.config().project_dir).name
    try:
        server = DashboardServer(cfg, provider, watch, gate, log=context.echo,
                                 actions=actions, local_project=local_project)
    except (OSError, ValueError) as exc:
        err_console.print(f"error: {exc}")
        raise typer.Exit(1) from exc
    server.serve_forever()


@dashboard_app.command("install")
def install_command(ctx: typer.Context) -> None:
    """Install and enable a systemd user service (auto-restart, survives reboot)."""
    context = get_context(ctx)
    project_dir = context.config().project_dir
    cfg = context.config().dashboard
    host, _ = cfg.listen_host_port
    try:
        info = svc.install_service(project_dir)
    except (JambuError, NotImplementedError) as exc:
        err_console.print(f"error: {exc}")
        raise typer.Exit(1) from exc
    new_token = None
    if host not in _LOOPBACK and _read_token(cfg, project_dir) is None:
        new_token = secrets.token_urlsafe(24)
        tf = _token_file(project_dir)
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(new_token)
        os.chmod(tf, 0o600)
    if context.json:
        print_json({**info, "token_generated": new_token is not None})
        return
    console.print(f"[green]+[/green] wrote {info['file']}")
    if new_token:
        console.print(f"[yellow]![/yellow] non-loopback bind: generated access token (shown once):")
        console.print(f"  [cyan]{new_token}[/cyan]")
        console.print(f"  stored at {_token_file(project_dir)} (chmod 600); reach the page with ?token=…")
    console.print("Enable it with:")
    console.print(f"  [cyan]{info['enable']}[/cyan]")


@dashboard_app.command("uninstall")
def uninstall_command(ctx: typer.Context) -> None:
    """Remove the systemd user service for this project."""
    context = get_context(ctx)
    info = svc.uninstall_service(context.config().project_dir)
    if context.json:
        print_json(info)
        return
    for path in info["removed"]:
        console.print(f"[red]-[/red] removed {path}")
    console.print("Disable it with:")
    console.print(f"  [cyan]{info['disable']}[/cyan]")


@dashboard_app.command("status")
def status_command(ctx: typer.Context) -> None:
    """Show whether the dashboard is installed and listening."""
    import http.client

    context = get_context(ctx)
    cfg = context.config().dashboard
    host, port = cfg.listen_host_port
    installed = svc.is_installed(context.config().project_dir)
    listening = False
    try:
        conn = http.client.HTTPConnection(host if host not in ("", "0.0.0.0") else "127.0.0.1", port, timeout=2)
        conn.request("HEAD", "/")
        listening = conn.getresponse().status < 500
        conn.close()
    except OSError:
        listening = False
    payload = {"installed": installed, "listening": listening, "listen": cfg.listen}
    if context.json:
        print_json(payload)
        return
    console.print(f"installed    {installed}")
    console.print(f"listening    {listening}  ({cfg.listen})")


@dashboard_app.command("logs")
def logs_command(ctx: typer.Context) -> None:
    """Print the journalctl command for the dashboard service (its logs)."""
    context = get_context(ctx)
    name = svc.unit_name(context.config().project_dir)
    console.print(f"  [cyan]journalctl --user -u {name}.service -f[/cyan]")
