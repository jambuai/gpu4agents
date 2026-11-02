"""Account-wide safety net: `gpu ls` and `gpu sweep`.

The watchdog guards one instance from the inside and `gpu guard` guards the
one instance a project tracks. The sweep covers every project on this machine
and whatever is left over:

- every project registered in ``$GPU_HOME/projects.json`` (by `gpu setup`)
  gets its own ``Engine.guard_tick``, with its own config and state: its own
  max_lifetime, lock, session and stopped_ttl. Those instances are *tracked*;
- an instance no project tracks is an *orphan*. Nobody knows its policy or
  whether it is in use, so the rules are blunt but never destructive for a
  running machine: a running orphan older than ``orphan_max_age`` is stopped
  (never destroyed); an orphan first seen stopped more than ``stopped_ttl``
  ago is destroyed.

Providers do not report when an instance stopped, so the first time the sweep
saw each orphan stopped is kept in ``$GPU_HOME/sweep.json``.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from . import events as ev
from . import policy
from .config import load_config
from .paths import gpu_home, project_state_dir
from .errors import JambuError, ProviderError
from .events import EventLog
from .models import Instance, InstanceState, is_jambu_label
from .state import RuntimeState, StateStore, from_iso, to_iso, utc_now

DEFAULT_ORPHAN_MAX_AGE_S = 24 * 3600.0
DEFAULT_STOPPED_TTL_S = 24 * 3600.0

REASON_ORPHAN_AGE = "orphan_max_age"
REASON_STOP_UNSUPPORTED = "stop_unsupported"

SWEEP_ACTION = "sweep_action"


def jambu_home() -> Path:
    return gpu_home()


class SweepRegistry:
    """First-seen-stopped timestamps, keyed by ``provider:instance_id``."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else jambu_home() / "sweep.json"
        self.stopped: dict[str, float] = {}
        self._load()

    @staticmethod
    def key(provider: str, instance_id: str) -> str:
        return f"{provider}:{instance_id}"

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text() or "{}")
        except (OSError, json.JSONDecodeError):
            # A corrupt registry only delays stopped_ttl; never block the sweep.
            return
        for key, value in (data.get("stopped") or {}).items():
            try:
                ts = from_iso(value)
            except Exception:  # noqa: BLE001 - one bad entry must not lose the rest
                continue
            if ts is not None:
                self.stopped[key] = ts

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "stopped": {k: to_iso(v) for k, v in sorted(self.stopped.items())},
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".sweep-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(json.dumps(payload, indent=2) + "\n")
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def first_seen_stopped(self, key: str, now: float) -> float:
        return self.stopped.setdefault(key, now)

    def forget(self, key: str) -> None:
        self.stopped.pop(key, None)

    def retain(self, provider: str, keys: set[str]) -> None:
        """Drop entries of ``provider`` that are no longer in the account."""
        prefix = f"{provider}:"
        for key in [k for k in self.stopped if k.startswith(prefix) and k not in keys]:
            del self.stopped[key]



def _atomic_json(path: Path, payload: dict, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class ProjectRegistry:
    """Absolute config paths of every project that provisioned from this machine.

    The sweep guards each with its own policy, so one project's thresholds are
    never applied to another project's instance.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else jambu_home() / "projects.json"

    def _read(self) -> dict[str, dict]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text() or "{}")
        except (OSError, json.JSONDecodeError):
            # A corrupt registry must not block the sweep; setup rewrites it.
            return {}
        projects = data.get("projects") if isinstance(data, dict) else None
        if not isinstance(projects, dict):
            return {}
        return {str(k): dict(v or {}) for k, v in projects.items()}

    @contextmanager
    def _edit(self) -> Iterator[dict[str, dict]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "a+") as lock:
            # Two projects running `gpu setup` at once must not drop each other.
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            projects = self._read()
            yield projects
            _atomic_json(self.path, {"version": 1, "projects": projects}, ".projects-")

    def paths(self) -> list[Path]:
        return [Path(p) for p in sorted(self._read())]

    def register(self, config_path: Path) -> Path:
        resolved = Path(config_path).expanduser().resolve()
        with self._edit() as projects:
            projects.setdefault(str(resolved), {"registered_at": to_iso(utc_now())})
        return resolved

    def remove(self, paths: list[Path]) -> None:
        if not paths:
            return
        with self._edit() as projects:
            for path in paths:
                projects.pop(str(path), None)


def register_project(config) -> Optional[Path]:
    """Register a loaded RuntimeConfig; None when it has no file behind it."""
    if getattr(config, "source_path", None) is None:
        return None
    return ProjectRegistry().register(config.source_path)


@dataclass
class ProjectGuard:
    """One project's own guard tick, run by the sweep."""

    config_path: Path
    provider: Optional[str] = None
    instance_id: Optional[str] = None
    decision: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": str(self.config_path),
            "provider": self.provider,
            "instance_id": self.instance_id,
            "decision": self.decision,
            "error": self.error,
        }


def _state_beside(config_path: Path) -> Optional[RuntimeState]:
    """The project's state read straight from disk (no API call)."""
    store = StateStore(project_state_dir(Path(config_path).parent))
    try:
        return store.load() if store.exists() else None
    except JambuError:
        return None


def guard_projects(
    paths: list[Path],
    make_engine: Callable[[Any], Any],
    *,
    apply: bool,
    echo=None,
) -> tuple[list[ProjectGuard], set[tuple[str, str]]]:
    """Run each project's own guard. Returns the results and the tracked ids.

    An instance whose project exists but could not be guarded (bad config, a
    busy lock, an API blip) is still tracked: not knowing its policy is no
    reason to treat it as an orphan.
    """
    say = echo or (lambda _m: None)
    results: list[ProjectGuard] = []
    tracked: set[tuple[str, str]] = set()
    for path in paths:
        guard = ProjectGuard(config_path=path)
        results.append(guard)
        try:
            config = load_config(path)
        except JambuError as exc:
            guard.error = f"config does not load: {exc}"
            state = _state_beside(path)
            if state is not None and state.instance_id:
                guard.provider, guard.instance_id = state.provider, state.instance_id
                tracked.add((state.provider, state.instance_id))
            say(f"! {path}: {guard.error}")
            continue
        try:
            engine = make_engine(config)
            guard.provider = engine.provider.name
            before = engine.store.load()
            if before.instance_id:
                guard.instance_id = before.instance_id
                tracked.add((guard.provider, before.instance_id))
            guard.decision = engine.guard_tick(apply=apply)
            after = engine.store.load()
            if after.instance_id:
                guard.instance_id = after.instance_id
                tracked.add((guard.provider, after.instance_id))
        except JambuError as exc:
            guard.error = str(exc)
            say(f"! {path}: guard failed: {exc}")
            continue
        _describe_guard(guard, apply, say)
    return results, tracked


def _describe_guard(guard: ProjectGuard, apply: bool, say) -> None:
    decision = guard.decision or {}
    action, reason = decision.get("action", policy.ACTION_NONE), decision.get("reason")
    if reason == "no_instance":
        say(f"= {guard.config_path}: no instance")
    elif action == policy.ACTION_NONE:
        say(f"= {guard.config_path}: {guard.instance_id} ok ({reason})")
    else:
        verb = action if apply else f"would {action}"
        say(f"- {guard.config_path}: {verb} {guard.instance_id} ({reason})")


@dataclass
class SweepAction:
    instance: Instance
    action: str  # stop | destroy | none
    reason: str
    age_seconds: Optional[float] = None
    stopped_seconds: Optional[float] = None
    applied: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance.id,
            "label": self.instance.label,
            "state": self.instance.state.value,
            "action": self.action,
            "reason": self.reason,
            "age_seconds": None if self.age_seconds is None else round(self.age_seconds, 1),
            "stopped_seconds": (
                None if self.stopped_seconds is None else round(self.stopped_seconds, 1)
            ),
            "applied": self.applied,
            "error": self.error,
        }


@dataclass
class SweepResult:
    provider: str
    apply: bool
    orphan_max_age_s: Optional[float]
    stopped_ttl_s: Optional[float]
    seen: int = 0
    tracked: int = 0
    actions: list[SweepAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "applied": self.apply,
            "orphan_max_age_s": self.orphan_max_age_s,
            "stopped_ttl_s": self.stopped_ttl_s,
            "instances_seen": self.seen,
            "tracked": self.tracked,
            "actions": [a.to_dict() for a in self.actions],
        }


def list_jambu_instances(provider) -> list[Instance]:
    provider.require_capability("list_instances")
    # The adapter already filters by label; filter again so a sloppy adapter
    # can never put someone else's machine in front of the sweep.
    return [i for i in provider.list_instances() if is_jambu_label(i.label)]


def sweep(
    provider,
    *,
    orphan_max_age_s: Optional[float],
    stopped_ttl_s: Optional[float],
    apply: bool = False,
    registry: Optional[SweepRegistry] = None,
    tracked: Optional[set[str]] = None,
    events: Optional[list[EventLog]] = None,
    echo=None,
    now: Optional[float] = None,
) -> SweepResult:
    """Plan (and with ``apply`` perform) the cleanup of untracked instances.

    ``tracked`` are the ids a project owns; they are left entirely to that
    project's own guard.
    """
    now = now if now is not None else utc_now()
    registry = registry or SweepRegistry()
    sinks = list(events or [])
    say = echo or (lambda _m: None)
    tracked = set(tracked or ())

    instances = list_jambu_instances(provider)
    result = SweepResult(
        provider=provider.name,
        apply=apply,
        orphan_max_age_s=orphan_max_age_s,
        stopped_ttl_s=stopped_ttl_s,
        seen=len(instances),
    )
    present: set[str] = set()

    for instance in instances:
        key = registry.key(provider.name, instance.id)
        if instance.id in tracked:
            # Its project measures stopped_ttl itself. Dropping our sighting
            # means the clock starts fresh should it ever become an orphan.
            result.tracked += 1
            continue
        present.add(key)

        if instance.state is InstanceState.STOPPED:
            stopped_for = now - registry.first_seen_stopped(key, now)
            if stopped_ttl_s is not None and stopped_for > stopped_ttl_s:
                result.actions.append(
                    SweepAction(
                        instance,
                        policy.ACTION_DESTROY,
                        policy.REASON_STOPPED_TTL,
                        stopped_seconds=stopped_for,
                    )
                )
            continue

        registry.forget(key)  # running again (or never stopped)
        if not instance.state.is_billable_compute or orphan_max_age_s is None:
            continue
        if not instance.created_at:
            continue  # no age to judge by; never guess on a running machine
        age = now - instance.created_at
        if age <= orphan_max_age_s:
            continue
        # Nobody knows whether a running orphan is in use, so it is only ever
        # stopped: the disk survives and stopped_ttl removes it later.
        if provider.capabilities.stop:
            action, reason = policy.ACTION_STOP, REASON_ORPHAN_AGE
        else:
            action, reason = policy.ACTION_NONE, REASON_STOP_UNSUPPORTED
        result.actions.append(SweepAction(instance, action, reason, age_seconds=age))

    registry.retain(provider.name, present)

    for planned in result.actions:
        _describe(planned, apply, say)
        if not apply or planned.action == policy.ACTION_NONE:
            continue
        try:
            if planned.action == policy.ACTION_DESTROY:
                provider.destroy(planned.instance)
                registry.forget(registry.key(provider.name, planned.instance.id))
            else:
                provider.stop(planned.instance)
            planned.applied = True
        except ProviderError as exc:
            planned.error = str(exc)
            say(f"! {planned.action} {planned.instance.id} failed: {exc}")
        for sink in sinks:
            _emit(sink, planned)

    registry.save()
    return result


def _describe(planned: SweepAction, apply: bool, say) -> None:
    from .durations import format_duration

    if planned.action == policy.ACTION_NONE:
        say(
            f"! orphan {planned.instance.id} ({planned.instance.label}) has run for "
            f"{format_duration(planned.age_seconds)}, but the provider cannot stop it"
        )
        return
    if planned.stopped_seconds is not None:
        why = f"stopped for {format_duration(planned.stopped_seconds)}"
    else:
        why = f"running for {format_duration(planned.age_seconds)}"
    verb = planned.action if apply else f"would {planned.action}"
    say(
        f"- {verb} orphan {planned.instance.id} ({planned.instance.label}): "
        f"{planned.reason}, {why}"
    )


def _emit(sink: EventLog, planned: SweepAction) -> None:
    detail = planned.to_dict()
    detail.pop("instance_id")
    sink.emit(SWEEP_ACTION, instance_id=planned.instance.id, source="sweep", **detail)
    if not planned.applied:
        return
    name = ev.INSTANCE_DESTROYED if planned.action == policy.ACTION_DESTROY else ev.INSTANCE_STOPPED
    sink.emit(name, instance_id=planned.instance.id, reason=planned.reason, source="sweep")


def tracked_by_projects(paths: list[Path]) -> dict[tuple[str, str], Path]:
    """(provider, instance id) -> config path, from state files only (no API calls)."""
    out: dict[tuple[str, str], Path] = {}
    for path in paths:
        state = _state_beside(path)
        if state is not None and state.instance_id and state.provider:
            out[(state.provider, state.instance_id)] = Path(path)
    return out


def account_rows(
    instances: list[Instance],
    tracked: dict[str, Path],
    now: Optional[float] = None,
) -> list[dict[str, Any]]:
    """``tracked`` maps instance id -> the config path of the project owning it."""
    now = now if now is not None else utc_now()
    return [
        {
            "instance_id": i.id,
            "label": i.label,
            "state": i.state.value,
            "gpu": f"{i.gpu_count}x {i.gpu_name}".strip() if i.gpu_name else "",
            "hourly_cost_usd": i.hourly_cost_usd,
            "created_at": i.created_at,
            "age_seconds": round(max(0.0, now - i.created_at), 1) if i.created_at else None,
            "tracked": i.id in tracked,
            "project": str(tracked[i.id]) if i.id in tracked else None,
        }
        for i in instances
    ]
