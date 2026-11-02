"""Structured lifecycle events (spec section 24).

Every lifecycle action appends one JSON object to ``.gpu/logs/events.jsonl``
so unexpected GPU cost can always be traced back to a decision and a reason.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from .state import to_iso, utc_now

# Minimum event vocabulary required by the spec.
PROVISION_REQUESTED = "provision_requested"
INSTANCE_CREATED = "instance_created"
RUNTIME_STARTING = "runtime_starting"
RUNTIME_READY = "runtime_ready"
WORKLOAD_STARTED = "workload_started"
WORKLOAD_FINISHED = "workload_finished"
INSTANCE_IDLE = "instance_idle"
INSTANCE_LOCKED = "instance_locked"
INSTANCE_UNLOCKED = "instance_unlocked"
INSTANCE_STOPPED = "instance_stopped"
INSTANCE_STARTED = "instance_started"
INSTANCE_DESTROYED = "instance_destroyed"
LIFECYCLE_GUARD_TRIGGERED = "lifecycle_guard_triggered"
PROVIDER_ERROR = "provider_error"
STATE_RECONCILED = "state_reconciled"
SETUP_FAILED = "setup_failed"
CLEANUP_PERFORMED = "cleanup_performed"
BUDGET_REJECTED = "budget_rejected"
# A machine failed to boot; setup rents another one.
HOST_FAILED = "host_failed"


class EventLog:
    """Append-only JSONL sink. Never raises into the caller's path."""

    def __init__(self, logs_dir: Path, echo: bool = False) -> None:
        self.dir = Path(logs_dir)
        self.path = self.dir / "events.jsonl"
        self.echo = echo
        self._lock = threading.Lock()
        self._context: dict[str, Any] = {}

    def bind(self, **context: Any) -> "EventLog":
        self._context.update({k: v for k, v in context.items() if v is not None})
        return self

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": to_iso(utc_now()),
            "event": event,
            "pid": os.getpid(),
        }
        record.update(self._context)
        record.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(record, default=str)
        with self._lock:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as handle:
                    handle.write(line + "\n")
            except OSError:
                pass
            if self.echo:
                print(line, file=sys.stderr)
        return record

    def tail(self, limit: int = 50, event: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event and record.get("event") != event:
                continue
            out.append(record)
        return out[-limit:]

    def extend(self, records: Iterable[dict[str, Any]]) -> int:
        """Merge records produced elsewhere (e.g. pulled from the remote watchdog)."""
        count = 0
        seen = {(r.get("ts"), r.get("event")) for r in self.tail(limit=2000)}
        with self._lock:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as handle:
                    for record in records:
                        key = (record.get("ts"), record.get("event"))
                        if key in seen:
                            continue
                        seen.add(key)
                        handle.write(json.dumps(record, default=str) + "\n")
                        count += 1
            except OSError:
                pass
        return count


class NullEventLog(EventLog):
    def __init__(self) -> None:  # noqa: D107
        super().__init__(Path(os.devnull).parent, echo=False)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:  # noqa: D102
        return {"event": event, **fields}
