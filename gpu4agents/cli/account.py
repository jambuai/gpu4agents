"""`gpu ls` / `gpu sweep` - every jambu instance in the account, not just this project's."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich.text import Text

from ..core import sweep as sweep_mod
from ..core.config import RuntimeConfig, find_config
from ..core.credentials import CredentialResolver
from ..core.durations import format_duration, parse_duration
from ..core.engine import Engine
from ..core.errors import ConfigError, JambuError
from ..core.events import EventLog
from ..providers.registry import load_builtin_providers
from . import timer
from ._common import Context, console, fmt_money, get_context, handle, print_json, state_text

DEFAULT_PROVIDER = "vast"


class Account:
    """A provider handle that works with or without a project around it."""

    def __init__(self, context: Context, provider_name: Optional[str] = None) -> None:
        self.config: Optional[RuntimeConfig] = None
        if context.config_path is not None or find_config() is not None:
            self.config = context.config()
        if self.config is not None:
            project_dir = self.config.project_dir
            profile = self.config.provider.profile
            name = provider_name or self.config.provider.name
            config = self.config
        else:
            # No project: a placeholder config only carries the provider name.
            project_dir = Path.cwd()
            profile = "default"
            name = provider_name or DEFAULT_PROVIDER
            config = RuntimeConfig.model_validate(
                {"provider": {"name": name}, "model": {"id": "jambu/account"}}
            )
        self.credentials = account_credentials(project_dir, profile)
        self.provider = make_provider(config, self.credentials, name)

    def project_paths(self) -> list[Path]:
        """Registered projects, plus the one this command runs in (ad hoc)."""
        paths = sweep_mod.ProjectRegistry().paths()
        if self.config is not None and self.config.source_path is not None:
            here = Path(self.config.source_path).resolve()
            if here not in paths:
                paths.append(here)
        return paths


def account_credentials(project_dir: Path, profile: str = "default") -> CredentialResolver:
    """The environment, ./.env, ~/.gpu/credentials, then registered projects' .env.

    The timer runs from ~ without the shell's environment; a key kept only in
    a project's .env must still reach the account-wide orphan sweep.
    """
    extra = [
        ("registered project .env", path.parent / ".env")
        for path in sweep_mod.ProjectRegistry().paths()
    ]
    return CredentialResolver(project_dir, profile=profile, extra_sources=extra)


def make_provider(config: RuntimeConfig, credentials: CredentialResolver, name: str):
    """Seam for tests: they swap in a FakeProvider here."""
    return load_builtin_providers().get(config, credentials, name=name)


def make_engine(config: RuntimeConfig, echo=None) -> Engine:
    """A project's own engine: its config, its credentials, its state."""
    credentials = CredentialResolver(config.project_dir, profile=config.provider.profile)
    provider = make_provider(config, credentials, config.provider.name)
    return Engine(config, provider=provider, echo=echo, credentials=credentials)


def ls_command(
    ctx: typer.Context,
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Provider to list (default: jambu.yaml's, else vast)."
    ),
) -> None:
    """List every jambu instance in the provider account."""
    context = get_context(ctx)
    try:
        account = Account(context, provider)
        instances = sweep_mod.list_jambu_instances(account.provider)
        owners = sweep_mod.tracked_by_projects(account.project_paths())
    except JambuError as exc:
        handle(exc)
        return
    name = account.provider.name
    tracked = {iid: path for (pname, iid), path in owners.items() if pname == name}
    rows = sweep_mod.account_rows(instances, tracked)

    if context.json:
        print_json({"provider": name, "instances": rows})
        return
    if not rows:
        console.print(Text(f"no jambu instances on {account.provider.display_name}", style="dim"))
        return
    table = Table(title=f"{account.provider.display_name} instances", title_justify="left", box=None)
    for column in ("ID", "LABEL", "STATE", "GPU", "$/H", "AGE", "PROJECT"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["instance_id"],
            row["label"],
            state_text(row["state"]),
            row["gpu"] or "-",
            fmt_money(row["hourly_cost_usd"]) or "-",
            format_duration(row["age_seconds"]),
            Text(row["project"], style="green") if row["tracked"] else Text("orphan", style="yellow"),
        )
    console.print(table)


def _seconds(flag: Optional[str], name: str) -> Optional[float]:
    try:
        return parse_duration(flag)
    except ValueError as exc:
        raise ConfigError(f"{name}: {exc}") from exc


def sweep_command(
    ctx: typer.Context,
    apply: bool = typer.Option(False, "--apply", help="Act. Without it the sweep only reports."),
    orphan_max_age: str = typer.Option(
        "24h", "--orphan-max-age", help="Stop an untracked running instance older than this."
    ),
    stopped_ttl: str = typer.Option(
        "24h", "--stopped-ttl", help="Destroy an untracked instance stopped longer than this."
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Provider to look for orphans on (default: jambu.yaml's, else vast)."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Also guard this project for this run, registered or not."
    ),
    install_timer: bool = typer.Option(
        False, "--install-timer", help="Write a systemd user timer that runs the sweep."
    ),
    every: str = typer.Option("5m", "--every", help="Timer period for --install-timer."),
    uninstall_timer: bool = typer.Option(
        False, "--uninstall-timer", help="Remove the systemd user timer."
    ),
) -> None:
    """Guard every registered project, then clean up orphans (dry run by default).

    Each project registered by `gpu setup` is guarded with its own jambu.yaml
    and state, exactly like `gpu guard` in that directory. An instance no
    project tracks is an orphan: it is stopped (never destroyed) once it has
    run longer than --orphan-max-age, and destroyed once it has been seen
    stopped longer than --stopped-ttl. Instances without a jambu- label are
    never touched.
    """
    context = get_context(ctx)
    if config is not None:
        context.config_path = config
    try:
        if install_timer or uninstall_timer:
            _timer(context, install_timer, every)
            return
        _sweep(context, apply, orphan_max_age, stopped_ttl, provider)
    except JambuError as exc:
        handle(exc)


def _require_file_credentials(context: Context) -> None:
    """The unit runs without the shell's environment: refuse what it cannot see."""
    checks: list[tuple[str, CredentialResolver, str]] = []
    if context.config_path is not None or find_config() is not None:
        cfg = context.config()
        checks.append(
            (
                cfg.provider.name,
                CredentialResolver(cfg.project_dir, profile=cfg.provider.profile),
                str(cfg.source_path),
            )
        )
    # The unit runs from ~ with no project: the orphan sweep's own lookup.
    checks.append((DEFAULT_PROVIDER, account_credentials(Path.home()), "the orphan sweep"))
    registry = load_builtin_providers()
    for provider_name, resolver, who in checks:
        for name in registry.get_class(provider_name).required_credentials:
            if resolver.file_source(name):
                continue
            where = "only in your shell environment" if resolver.get(name) else "not set anywhere"
            raise ConfigError(
                f"{name} is {where}, so the timer could not use it for {who}: a systemd "
                f"user unit does not inherit your shell. Put `{name}=...` in "
                "~/.gpu/credentials (chmod 600), then re-run --install-timer."
            )


def _timer(context: Context, install: bool, every: str) -> None:
    if not install:
        result = timer.uninstall_timer()
        if context.json:
            print_json(result)
            return
        for path in result["removed"]:
            console.print(Text(f"- removed {path}", style="red"))
        if not result["removed"]:
            console.print(Text("= no timer installed", style="dim"))
        console.print(f"  then run: {result['disable']}")
        return

    every_s = _seconds(every, "--every")
    if not every_s:
        raise ConfigError("--every must be a positive duration, e.g. 5m")
    registry = sweep_mod.ProjectRegistry()
    project = None
    if context.config_path is not None or find_config() is not None:
        project = registry.register(context.config().source_path)
    _require_file_credentials(context)
    result = timer.install_timer(every_s)
    # Units written before the registry carried one guard line per project.
    for legacy in result["legacy_guard_configs"]:
        if Path(legacy).is_file():
            registry.register(Path(legacy))
    result["project_config"] = str(project) if project else None
    result["projects"] = [str(p) for p in registry.paths()]
    if context.json:
        print_json(result)
        return
    console.print(Text(f"+ wrote {result['service']}", style="green"))
    console.print(Text(f"+ wrote {result['timer']}", style="green"))
    if project:
        console.print(Text(f"+ registered {project}", style="green"))
    console.print(
        f"  every {format_duration(result['every_s'])}: `gpu sweep --apply`, guarding "
        f"{len(result['projects'])} registered project(s) and cleaning up orphans"
    )
    console.print(Text(f"> enable it: {result['enable']}", style="cyan"))
    console.print(
        Text(
            "  user timers stop at logout unless lingering is on: loginctl enable-linger $USER",
            style="dim",
        )
    )


def _sweep(
    context: Context,
    apply: bool,
    orphan_max_age: str,
    stopped_ttl: str,
    provider_name: Optional[str],
) -> None:
    orphan_max_age_s = _seconds(orphan_max_age, "--orphan-max-age")
    stopped_ttl_s = _seconds(stopped_ttl, "--stopped-ttl")
    account = Account(context, provider_name)
    registry = sweep_mod.ProjectRegistry()
    registered = registry.paths()
    # A project whose jambu.yaml is gone has nothing left to guard it by.
    gone = [p for p in registered if not p.is_file()]
    registry.remove(gone)
    for path in gone:
        context.echo(f"- unregistered {path} (config file is gone)")
    paths = [p for p in account.project_paths() if p not in gone]

    if not context.json and paths:
        context.echo(f"> guarding {len(paths)} project(s)" + ("" if apply else " (dry run)"))
    guards, owned = sweep_mod.guard_projects(
        paths, lambda cfg: make_engine(cfg, echo=context.echo), apply=apply, echo=context.echo
    )
    name = account.provider.name
    tracked = {iid for (pname, iid) in owned if pname == name}

    if not context.json:
        context.echo(
            f"> sweeping {account.provider.display_name} for orphans: running > "
            f"{format_duration(orphan_max_age_s) if orphan_max_age_s else 'never'} is stopped, "
            f"stopped > {format_duration(stopped_ttl_s) if stopped_ttl_s else 'never'} is destroyed"
            + ("" if apply else " (dry run)")
        )
    try:
        result = sweep_mod.sweep(
            account.provider,
            orphan_max_age_s=orphan_max_age_s,
            stopped_ttl_s=stopped_ttl_s,
            apply=apply,
            tracked=tracked,
            events=[EventLog(sweep_mod.jambu_home() / "logs").bind(provider=name)],
            echo=context.echo,
        )
    except JambuError as exc:
        # The projects were guarded already; report that, then fail the run.
        if context.json:
            print_json(
                {
                    "provider": name,
                    "error": str(exc),
                    "projects": [g.to_dict() for g in guards],
                    "unregistered": [str(p) for p in gone],
                }
            )
            raise typer.Exit(getattr(exc, "exit_code", 1)) from exc
        raise
    failed = any(a.error for a in result.actions) or any(g.error for g in guards)
    if context.json:
        payload = result.to_dict()
        payload["projects"] = [g.to_dict() for g in guards]
        payload["unregistered"] = [str(p) for p in gone]
        print_json(payload)
        if failed:
            raise typer.Exit(1)
        return
    pending = [a for a in result.actions if a.action != "none"] + [
        g for g in guards if (g.decision or {}).get("action", "none") != "none"
    ]
    if not pending:
        console.print(
            Text(
                f"= {result.seen} jambu instance(s), {result.tracked} tracked, nothing to do",
                style="green",
            )
        )
    elif not apply:
        console.print(Text(f"! {len(pending)} action(s) pending; re-run with --apply", style="yellow"))
    if failed:
        raise typer.Exit(1)
