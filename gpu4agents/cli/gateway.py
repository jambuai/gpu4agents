"""`gpu gateway`: a managed local endpoint that follows the rented machine.

    gpu gateway start        run the gateway in the foreground (what the unit runs)
    gpu gateway install      install + enable a systemd user service (auto-restart, boot-safe)
    gpu gateway uninstall    remove the service
    gpu gateway status       is it installed / listening, and where is it pointing
    gpu gateway logs         journalctl for the service

The gateway reads only this project's gpu.yaml and .gpu/state.json, so the
service needs no credentials in its environment.
"""

from __future__ import annotations

import typer

from ..core.errors import JambuError
from ..gateway import EndpointSource, GatewayServer, adapter_names, get_adapter
from ..gateway import service as svc
from ._common import console, err_console, get_context, print_json

gateway_app = typer.Typer(help="Local gateway: a stable loopback endpoint that follows the machine.")


def _conn_provider(context):
    """A callable the server polls for the current upstream connection."""
    engine = context.engine()

    def provider() -> dict:
        return engine.connection(engine.store.load())

    return provider


@gateway_app.command("start")
def start_command(ctx: typer.Context) -> None:
    """Run the gateway in the foreground (blocks). This is what the service runs."""
    context = get_context(ctx)
    try:
        cfg = context.config().gateway
        adapter = get_adapter(cfg)
        host, port = cfg.listen_host_port
    except (JambuError, ValueError) as exc:
        err_console.print(f"error: {exc}")
        raise typer.Exit(getattr(exc, "exit_code", 2)) from exc
    if not cfg.enabled:
        context.echo("! gateway.enabled is false in gpu.yaml; starting anyway because you asked")
    source = EndpointSource(_conn_provider(context), ttl=max(5.0, cfg.options.get("refresh_s", 60.0)))
    try:
        server = GatewayServer(cfg, adapter, source, log=context.echo)
    except OSError as exc:
        err_console.print(f"error: cannot bind {host}:{port}: {exc}")
        raise typer.Exit(1) from exc
    server.serve_forever()


@gateway_app.command("install")
def install_command(ctx: typer.Context) -> None:
    """Install and enable a systemd user service (auto-restart, survives reboot)."""
    context = get_context(ctx)
    project_dir = context.config().project_dir
    try:
        info = svc.install_service(project_dir)
    except (JambuError, NotImplementedError) as exc:
        err_console.print(f"error: {exc}")
        raise typer.Exit(1) from exc
    if context.json:
        print_json(info)
        return
    console.print(f"[green]+[/green] wrote {info['file']}")
    console.print("Enable it with:")
    console.print(f"  [cyan]{info['enable']}[/cyan]")


@gateway_app.command("uninstall")
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


@gateway_app.command("status")
def status_command(ctx: typer.Context) -> None:
    """Show whether the gateway is installed, listening, and where it points."""
    import http.client

    context = get_context(ctx)
    cfg = context.config().gateway
    project_dir = context.config().project_dir
    host, port = cfg.listen_host_port
    listening = False
    try:
        conn = http.client.HTTPConnection(host, port, timeout=2)
        conn.request("HEAD", "/")
        listening = conn.getresponse().status == 200
        conn.close()
    except OSError:
        listening = False
    upstream = None
    try:
        engine = context.engine()
        upstream = engine.connection(engine.store.load()).get("endpoint")
    except JambuError:
        upstream = None
    payload = {
        "adapter": cfg.adapter,
        "listen": f"{host}:{port}",
        "installed": svc.is_installed(project_dir),
        "listening": listening,
        "upstream_endpoint": upstream,
        "adapters_available": adapter_names(),
    }
    if context.json:
        print_json(payload)
        return
    console.print(f"adapter      {payload['adapter']}")
    console.print(f"listen       http://{host}:{port}")
    console.print(f"installed    {payload['installed']}")
    console.print(f"listening    {payload['listening']}")
    console.print(f"upstream     {upstream or '(none - run gpu setup)'}")


@gateway_app.command("logs")
def logs_command(ctx: typer.Context) -> None:
    """Print the journalctl command for the gateway service (its logs)."""
    context = get_context(ctx)
    name = svc.unit_name(context.config().project_dir)
    console.print(f"  [cyan]journalctl --user -u {name}.service -f[/cyan]")
