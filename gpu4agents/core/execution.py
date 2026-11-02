"""Workload execution and heartbeat (spec sections 12, 13, 16).

The CLI registers the workload with the remote watchdog, keeps a heartbeat
alive while it runs, and records completion. If the heartbeat disappears
(CLI crash, SSH drop, laptop sleep), the watchdog lets the grace period
expire and the instance falls back to IDLE and the normal shutdown rules.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Optional, Sequence

from .beacon import WatchdogClient
from .events import WORKLOAD_FINISHED, WORKLOAD_STARTED, EventLog
from .models import Execution, Workload
from .state import StateStore, WorkloadRecord, new_workload_id, utc_now


class Heartbeat:
    """Background thread that keeps the remote watchdog informed."""

    def __init__(
        self,
        workload_id: str,
        interval: float,
        store: StateStore,
        beacon: Optional[WatchdogClient] = None,
    ) -> None:
        self.workload_id = workload_id
        self.interval = max(5.0, float(interval))
        self.store = store
        self.beacon = beacon
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.failures = 0

    def _beat(self) -> None:
        if self.beacon is not None:
            if self.beacon.workload_heartbeat(self.workload_id) is None:
                self.failures += 1
            else:
                self.failures = 0
        try:
            with self.store.transaction(timeout=10) as state:
                record = state.workloads.get(self.workload_id)
                if record is not None:
                    record.last_heartbeat = utc_now()
                state.touch()
        except Exception:  # noqa: BLE001 - a heartbeat must never kill a workload
            pass

    def start(self) -> "Heartbeat":
        self._beat()

        def _loop() -> None:
            while not self._stop.wait(self.interval):
                self._beat()

        self._thread = threading.Thread(target=_loop, name="jambu-heartbeat", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "Heartbeat":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()


class ExecutionManager:
    """Registers, runs and closes out workloads."""

    def __init__(
        self,
        store: StateStore,
        events: EventLog,
        beacon: Optional[WatchdogClient] = None,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self.store = store
        self.events = events
        self.beacon = beacon
        self.heartbeat_interval = heartbeat_interval

    def register(self, command: Sequence[str], label: str = "") -> WorkloadRecord:
        record = WorkloadRecord(
            id=new_workload_id(),
            label=label or " ".join(command)[:80],
            command=list(command),
            state="running",
            pid=os.getpid(),
        )
        with self.store.transaction() as state:
            state.workloads[record.id] = record
            state.touch()
            state.prune_workloads()
        if self.beacon is not None:
            self.beacon.workload_start(record.id, record.label)
        self.events.emit(WORKLOAD_STARTED, workload_id=record.id, label=record.label)
        return record

    def finish(self, record: WorkloadRecord, returncode: Optional[int]) -> None:
        state_name = "finished" if returncode == 0 else "failed"
        with self.store.transaction() as state:
            stored = state.workloads.get(record.id)
            if stored is not None:
                stored.state = state_name
                stored.finished_at = utc_now()
                stored.returncode = returncode
            state.touch()
        if self.beacon is not None:
            self.beacon.workload_end(record.id, returncode, state_name)
        self.events.emit(
            WORKLOAD_FINISHED,
            workload_id=record.id,
            returncode=returncode,
            state=state_name,
            duration_seconds=round(utc_now() - record.started_at, 1),
        )

    def run_local(self, workload: Workload, record: WorkloadRecord) -> Execution:
        """Run the command on the developer machine against the remote runtime."""
        execution = Execution(
            id=record.id, workload_id=record.id, started_at=time.time()
        )
        env = os.environ.copy()
        env.update({k: str(v) for k, v in workload.env.items()})

        process = subprocess.Popen(  # noqa: S603 - the user's own command, by design
            list(workload.command),
            cwd=workload.cwd or None,
            env=env,
        )

        def _forward(signum, _frame):  # pragma: no cover - interactive path
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass

        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.signal(sig, _forward)
            except ValueError:
                pass
        try:
            returncode = process.wait()
        finally:
            for sig, handler in previous.items():
                try:
                    signal.signal(sig, handler)
                except ValueError:
                    pass

        execution.finished_at = time.time()
        execution.returncode = returncode
        execution.state = "finished" if returncode == 0 else "failed"
        return execution
