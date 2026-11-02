"""Observed runtime state (spec section 9).

jambu.yaml is the desired-state SoT; ``.gpu/state.json`` is what we last
observed. This module owns reading/writing it atomically, plus the on-disk
process lock that keeps two CLI invocations from provisioning twice.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .errors import LockBusy, StateError
from .models import Instance, InstanceState
from .policy import Workload as PolicyWorkload

STATE_VERSION = 1


def utc_now() -> float:
    return time.time()


def to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def from_iso(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError as exc:
        raise StateError(f"invalid timestamp in state file: {value!r}") from exc


@dataclass
class WorkloadRecord:
    id: str
    label: str = ""
    command: list[str] = field(default_factory=list)
    state: str = "running"  # running | queued | finished | failed
    started_at: float = field(default_factory=utc_now)
    last_heartbeat: float = field(default_factory=utc_now)
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    pid: Optional[int] = None

    def to_policy(self) -> PolicyWorkload:
        return PolicyWorkload(
            id=self.id,
            started_at=self.started_at,
            last_heartbeat=self.last_heartbeat,
            label=self.label,
            state=self.state,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["started_at"] = to_iso(self.started_at)
        data["last_heartbeat"] = to_iso(self.last_heartbeat)
        data["finished_at"] = to_iso(self.finished_at)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkloadRecord":
        return cls(
            id=data["id"],
            label=data.get("label", ""),
            command=list(data.get("command") or []),
            state=data.get("state", "running"),
            started_at=from_iso(data.get("started_at")) or utc_now(),
            last_heartbeat=from_iso(data.get("last_heartbeat")) or utc_now(),
            finished_at=from_iso(data.get("finished_at")),
            returncode=data.get("returncode"),
            pid=data.get("pid"),
        )


@dataclass
class WatchdogInfo:
    installed: bool = False
    url: Optional[str] = None
    token: Optional[str] = None
    port: Optional[int] = None
    last_contact: Optional[float] = None
    last_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["last_contact"] = to_iso(self.last_contact)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchdogInfo":
        return cls(
            installed=bool(data.get("installed")),
            url=data.get("url"),
            token=data.get("token"),
            port=data.get("port"),
            last_contact=from_iso(data.get("last_contact")),
            last_error=data.get("last_error"),
        )


@dataclass
class RuntimeState:
    provider: str = ""
    instance_id: Optional[str] = None
    fingerprint: Optional[str] = None
    state: str = InstanceState.DESTROYED.value
    created_at: Optional[float] = None
    last_activity: Optional[float] = None
    locked: bool = False
    locked_until: Optional[float] = None
    lock_reason: str = ""
    starting: bool = False
    model_id: Optional[str] = None
    endpoint: Optional[str] = None
    hourly_cost_usd: Optional[float] = None
    session_cost_usd: float = 0.0
    # Billable-time bookkeeping (spec section 25): session_cost_usd must
    # reflect actual running time, not wall-clock time since creation - a
    # stopped instance costs nothing in compute-hours even if `status` is
    # checked hours later. `billed_seconds` accumulates completed billable
    # periods (e.g. before a stop); `running_since` marks an open one.
    billed_seconds: float = 0.0
    running_since: Optional[float] = None
    # A session is one continuous running period of the instance: it starts
    # at creation and again at every restart. max_lifetime and the session
    # budget are measured from here, so a restarted instance gets a fresh
    # bound instead of being killed for the hours it spent stopped.
    session_started_at: Optional[float] = None
    # When the instance was first observed STOPPED; drives lifecycle.stopped_ttl
    # on providers where a stopped instance still bills (storage).
    stopped_since: Optional[float] = None
    # The key the model server demands (vLLM --api-key): runtime.api_key, or
    # one generated per instance so a public endpoint is never open to anyone.
    api_key: Optional[str] = None
    instance: Optional[Instance] = None
    workloads: dict[str, WorkloadRecord] = field(default_factory=dict)
    watchdog: WatchdogInfo = field(default_factory=WatchdogInfo)

    # -- helpers ------------------------------------------------------------

    def has_instance(self) -> bool:
        return bool(self.instance_id) and self.state != InstanceState.DESTROYED.value

    def touch(self) -> None:
        self.last_activity = utc_now()

    def accrued_billable_seconds(self, now: Optional[float] = None) -> float:
        """Total seconds actually spent in a billable-compute state.

        Unlike ``now - created_at``, this excludes time spent stopped -
        checking `status` long after an instance auto-stopped must not keep
        inflating the reported cost.
        """
        now = now if now is not None else utc_now()
        open_period = max(0.0, now - self.running_since) if self.running_since is not None else 0.0
        return self.billed_seconds + open_period

    def mark_billable(self, is_billable: bool, now: Optional[float] = None) -> None:
        """Record a transition into/out of a billable-compute state.

        Call this every time reconciliation observes the current state -
        it's a no-op unless billable-ness actually changed since the last call.
        """
        now = now if now is not None else utc_now()
        was_billable = self.running_since is not None
        if is_billable and not was_billable:
            self.running_since = now
        elif not is_billable and was_billable:
            self.billed_seconds += max(0.0, now - self.running_since)
            self.running_since = None

    def mark_stopped(self, now: Optional[float] = None, ran_at: Iterable[Optional[float]] = ()) -> None:
        """Record that the instance is (still) stopped; keeps the earliest time.

        ``ran_at`` are times the instance was evidently running. One newer than
        ``stopped_since`` means it ran (and stopped again) while this CLI was
        not looking, so the stop clock restarts instead of destroying a fresh stop.
        """
        now = now if now is not None else utc_now()
        if self.stopped_since is None:
            self.stopped_since = now
            return
        evidence = [
            self.session_started_at,
            self.last_activity,
            self.watchdog.last_contact,
            *ran_at,
        ]
        # The state file keeps whole seconds; a sub-second lead is rounding.
        if any(ts is not None and ts > self.stopped_since + 1.0 for ts in evidence):
            self.stopped_since = now

    def mark_session_start(self, now: Optional[float] = None) -> None:
        self.session_started_at = now if now is not None else utc_now()
        self.stopped_since = None

    def pending_workloads(self) -> list[WorkloadRecord]:
        return [w for w in self.workloads.values() if w.state in ("running", "queued")]

    def prune_workloads(self, keep: int = 25) -> None:
        finished = sorted(
            (w for w in self.workloads.values() if w.state not in ("running", "queued")),
            key=lambda w: w.finished_at or w.started_at,
        )
        for record in finished[:-keep] if len(finished) > keep else []:
            self.workloads.pop(record.id, None)

    def clear_instance(self, state: str = InstanceState.DESTROYED.value) -> None:
        # Close out any open billable period and freeze the final total into
        # session_cost_usd (the historical record of the finished session)
        # before resetting the counters for whatever instance comes next.
        self.mark_billable(False)
        if self.hourly_cost_usd:
            self.session_cost_usd = round(self.hourly_cost_usd * self.billed_seconds / 3600.0, 4)
        self.instance_id = None
        self.instance = None
        self.endpoint = None
        self.api_key = None
        self.state = state
        self.starting = False
        self.locked = False
        self.locked_until = None
        self.watchdog = WatchdogInfo()
        self.billed_seconds = 0.0
        self.running_since = None
        self.session_started_at = None
        self.stopped_since = None
        for record in self.workloads.values():
            if record.state in ("running", "queued"):
                record.state = "failed"
                record.finished_at = utc_now()

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "provider": self.provider,
            "instance_id": self.instance_id,
            "fingerprint": self.fingerprint,
            "state": self.state,
            "created_at": to_iso(self.created_at),
            "last_activity": to_iso(self.last_activity),
            "locked": self.locked,
            "locked_until": to_iso(self.locked_until),
            "lock_reason": self.lock_reason,
            "starting": self.starting,
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "hourly_cost_usd": self.hourly_cost_usd,
            "session_cost_usd": round(self.session_cost_usd, 4),
            "billed_seconds": round(self.billed_seconds, 1),
            "running_since": to_iso(self.running_since),
            "session_started_at": to_iso(self.session_started_at),
            "stopped_since": to_iso(self.stopped_since),
            "api_key": self.api_key,
            "instance": self.instance.to_dict() if self.instance else None,
            "workloads": {k: v.to_dict() for k, v in self.workloads.items()},
            "watchdog": self.watchdog.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeState":
        instance_data = data.get("instance")
        return cls(
            provider=data.get("provider", ""),
            instance_id=data.get("instance_id"),
            fingerprint=data.get("fingerprint"),
            state=data.get("state", InstanceState.DESTROYED.value),
            created_at=from_iso(data.get("created_at")),
            last_activity=from_iso(data.get("last_activity")),
            locked=bool(data.get("locked")),
            locked_until=from_iso(data.get("locked_until")),
            lock_reason=data.get("lock_reason", ""),
            starting=bool(data.get("starting")),
            model_id=data.get("model_id"),
            endpoint=data.get("endpoint"),
            hourly_cost_usd=data.get("hourly_cost_usd"),
            session_cost_usd=float(data.get("session_cost_usd") or 0.0),
            billed_seconds=float(data.get("billed_seconds") or 0.0),
            running_since=from_iso(data.get("running_since")),
            session_started_at=from_iso(data.get("session_started_at")),
            stopped_since=from_iso(data.get("stopped_since")),
            api_key=data.get("api_key"),
            instance=Instance.from_dict(instance_data) if instance_data else None,
            workloads={
                k: WorkloadRecord.from_dict(v) for k, v in (data.get("workloads") or {}).items()
            },
            watchdog=WatchdogInfo.from_dict(data.get("watchdog") or {}),
        )


class StateStore:
    """Atomic reader/writer for ``.gpu/state.json``."""

    def __init__(self, state_dir: Path) -> None:
        self._depth = 0  # flock is per-open-file-description: keep it reentrant
        self.dir = Path(state_dir)
        self.path = self.dir / "state.json"
        self.lock_path = self.dir / "runtime.lock"
        self.logs_dir = self.dir / "logs"

    def ensure_dirs(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        gitignore = self.dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> RuntimeState:
        if not self.path.is_file():
            return RuntimeState()
        try:
            data = json.loads(self.path.read_text() or "{}")
        except json.JSONDecodeError as exc:
            raise StateError(f"{self.path} is corrupt: {exc}") from exc
        if not data:
            return RuntimeState()
        return RuntimeState.from_dict(data)

    def save(self, state: RuntimeState) -> None:
        self.ensure_dirs()
        payload = json.dumps(state.to_dict(), indent=2, sort_keys=False) + "\n"
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix=".state-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @contextmanager
    def transaction(self, timeout: float = 30.0) -> Iterator[RuntimeState]:
        """Exclusive read-modify-write. Guarantees `setup` cannot race itself."""
        with self.process_lock(timeout=timeout):
            state = self.load()
            yield state
            self.save(state)

    @contextmanager
    def process_lock(self, timeout: float = 30.0) -> Iterator[None]:
        if self._depth > 0:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self.ensure_dirs()
        handle = open(self.lock_path, "a+")
        deadline = time.time() + timeout
        try:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() >= deadline:
                        raise LockBusy(
                            "another gpu command is holding the runtime lock "
                            f"({self.lock_path}); retry when it finishes"
                        ) from None
                    time.sleep(0.2)
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()} {to_iso(utc_now())}\n")
            handle.flush()
            self._depth = 1
            yield
        finally:
            self._depth = 0
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def new_workload_id() -> str:
    return uuid.uuid4().hex[:12]
