"""The coordinator: the loop that turns queued jobs into delivered results.

One coordinator is active per queue (a lease in the backend); any other waits as a
standby and takes over when the lease runs out. Each tick, in order:

1. renew the coordinator lease, or try to acquire it (on acquire: ``recover()``);
2. ``pool.poll()``;
3. renew the lease of every job in hand;
4. machine faults first (gone, unresponsive, unhealthy: retire, requeue what ran
   there), then every running attempt: finished -> deliver (fetch, health check,
   store, verify, settle), cancel requested -> cancel on the machine, settle;
5. while a ready machine has a free slot and the queue is not paused: claim,
   preflight, push the machine's deadline, dispatch ``<job>-a<n>``;
6. the scaling decision: open machines, keep ``min_machines`` warm;
7. the audit hook every 12 ticks (Task 12);
8. metrics.

What is in flight is in the queue, not only in this process: before a dispatch the
job's ``limits`` record the attempt number, remote id and slot. A coordinator that
finds such a job (after a crash, a takeover or an expired lease) looks for that
remote id on the machines and adopts the run instead of starting it twice.

Writes are fenced by the claim token: a ``LeaseLost`` drops the job from memory, is
reported as a stale write, and the write is never retried.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import socket
import tarfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..core import events as ev
from ..core.config import QueueConfig, RuntimeConfig
from ..core.durations import parse_duration
from .backend import QueueBackend
from .models import FINAL_STATES, Attempt, Claim, JobState, LeaseLost, Outcome, QueueJob
from .observe import Observer
from .outputs import OutputRef, OutputStore
from .pool_types import MachineState, PoolMachine
from .preflight import Limits, PreflightContext, load_preflight
from .scaler import MachineView, ScaleInput, decide
from .spend import pool_usd_per_hour

AUDIT_EVERY_TICKS = 12
COORDINATOR_LEASE_TICKS = 3
JOB_LEASE_TICKS = 12
# A tick can block on one health check (its timeout plus the pool's slack): no lease
# may run out inside one.
HEALTH_SLACK_S = 30.0
PREVIOUS_DIR = ".jambu-previous"
# The watchdog's job runner default (JAMBU_JOBS_DIR); the command also exports the
# real path, so a machine with another jobs directory still gets it right.
REMOTE_JOBS_DIR = "/workspace/jobs"

_LIVE = (MachineState.READY, MachineState.BUSY)
_ANSWERING = (MachineState.READY, MachineState.BUSY, MachineState.DRAINING, MachineState.HELD)
_FAULTS = {
    MachineState.GONE: (Outcome.MACHINE_LOST, "machine gone", False),
    MachineState.UNRESPONSIVE: (Outcome.UNRESPONSIVE, "unresponsive", True),
    MachineState.UNHEALTHY: (Outcome.UNHEALTHY, "unhealthy", True),
}
_JOB_TERMINAL = (Outcome.SUCCEEDED, Outcome.JOB_FAILED, Outcome.TIMEOUT)


@dataclass
class TickReport:
    claimed: int
    dispatched: int
    settled: int
    requeued: int
    opened: int
    violations: int


class _Dropped(Exception):
    """The job left this coordinator's hands (stale lease); stop working on it."""


class _Demoted(Exception):
    """This coordinator lost its lease in the middle of a tick."""


@dataclass
class _Run:
    """A job in hand: claimed, then pending -> running (or cancelling) -> delivering."""

    claim: Claim
    number: int
    remote_id: str
    phase: str = "pending"
    slot: Optional[str] = None  # set once a dispatch was tried: retries go to the same machine
    limits: Optional[Limits] = None
    started_at: Optional[float] = None
    locate: bool = False  # an earlier coordinator may have started it: find it first
    exit_code: Optional[int] = None
    remote_state: Optional[str] = None
    fetched: Optional[Path] = None
    checked: bool = False  # the health check after the run is done
    delivering_since: Optional[float] = None

    @property
    def job(self) -> QueueJob:
        return self.claim.job

    @property
    def job_id(self) -> str:
        return self.claim.job.id


def _default_worker() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(3)}"


def _limits_from(lim: dict) -> Optional[Limits]:
    try:
        return Limits(
            timeout_s=float(lim["timeout_s"]),
            max_cost_usd=lim.get("max_cost_usd"),
            deadline_margin_s=float(lim.get("deadline_margin_s") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


class Coordinator:
    def __init__(
        self,
        config: RuntimeConfig,
        backend: QueueBackend,
        outputs: OutputStore,
        pool: Any,
        observer: Observer,
        preflight: Optional[Callable] = None,
        worker_id: Optional[str] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.queue: QueueConfig = config.queue or QueueConfig()
        self.backend = backend
        self.outputs = outputs
        self.pool = pool
        self.obs = observer
        self.preflight = preflight or load_preflight(self.queue.preflight)
        self.worker_id = worker_id or _default_worker()
        self.clock = clock

        self.tick_s = self.queue.tick_s
        slow_s = (parse_duration(self.queue.health_check.timeout) or 120.0) + HEALTH_SLACK_S
        self.coordinator_lease_s = max(COORDINATOR_LEASE_TICKS * self.tick_s, slow_s + 2 * self.tick_s)
        self.job_lease_s = max(JOB_LEASE_TICKS * self.tick_s, slow_s + 2 * self.tick_s)
        self.workdir = Path(config.state_dir) / "queue" / "work"

        self.leader = False
        self._runs: dict[str, _Run] = {}
        self._adopt: set[str] = set()  # in-flight jobs whose old lease has not run out yet
        self._renewed_at: Optional[float] = None
        self._ticks = 0
        self._last_decision: Optional[tuple] = None
        self._last_blocked: Optional[str] = None
        self._stats = None
        self._report = TickReport(0, 0, 0, 0, 0, 0)
        # Task 12 wires the auditor here: called every AUDIT_EVERY_TICKS ticks, returns violations.
        self._audit: Optional[Callable[[], Any]] = None

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def config_hash(self) -> str:
        body = json.dumps(self.queue.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def run(self, stop: threading.Event) -> None:
        """Tick every ``queue.tick`` until ``stop`` is set; standby while another
        coordinator holds the lease. Releases the lease on the way out."""
        self.obs.emit(
            ev.COORDINATOR_STARTED, worker=self.worker_id, config_hash=self.config_hash(),
            backend=getattr(self.backend, "name", None), tick_s=self.tick_s,
        )
        try:
            while not stop.is_set():
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 - the loop outlives one bad tick
                    self.obs.emit(
                        ev.PROVIDER_ERROR, operation="coordinator_tick",
                        error_type=type(exc).__name__, error=str(exc)[:500],
                    )
                stop.wait(self.tick_s)
        finally:
            if self.leader:
                try:
                    self.backend.release_coordinator(self.worker_id)
                except Exception:  # noqa: BLE001 - the lease runs out by itself
                    pass
            self.leader = False
            self.obs.emit(ev.COORDINATOR_STOPPED, worker=self.worker_id)

    def tick(self) -> TickReport:
        report = self._report = TickReport(0, 0, 0, 0, 0, 0)
        try:
            if not self._lead():
                self._metrics()
                return report
            try:
                self.pool.poll()
            except Exception as exc:  # noqa: BLE001
                self.obs.emit(ev.PROVIDER_ERROR, operation="pool_poll", error=str(exc)[:500])
            self._renew_jobs()
            self._machine_faults()
            self._watch()
            self._place()
            self._scale()
        except _Demoted:
            pass
        self._ticks += 1
        if self._audit is not None and self.leader and self._ticks % AUDIT_EVERY_TICKS == 0:
            try:
                found = self._audit()
                report.violations = len(found or [])
            except Exception as exc:  # noqa: BLE001
                self.obs.emit(ev.PROVIDER_ERROR, operation="audit", error=str(exc)[:500])
        self._metrics()
        return report

    # ------------------------------------------------------------------
    # leases
    # ------------------------------------------------------------------

    def _lead(self) -> bool:
        if self.leader:
            if self.backend.renew_coordinator(self.worker_id, self.coordinator_lease_s):
                self._renewed_at = self.clock()
                return True
            self._demote()
        if not self.backend.acquire_coordinator(self.worker_id, self.coordinator_lease_s):
            return False
        self.leader = True
        self._renewed_at = self.clock()
        self.obs.emit(ev.COORDINATOR_LEASE_ACQUIRED, worker=self.worker_id, lease_s=self.coordinator_lease_s)
        self.recover()
        return True

    def _demote(self) -> None:
        self.leader = False
        self.obs.inc("gpu_queue_lease_renew_failures_total")
        self.obs.emit(ev.COORDINATOR_LEASE_LOST, worker=self.worker_id, jobs_dropped=sorted(self._runs))
        # Whoever holds the lease now adopts these; nothing is written for them from here.
        for run in list(self._runs.values()):
            self._drop(run)
        self._adopt.clear()

    def _heartbeat(self) -> None:
        """Renew every lease when a tick has run long (uploads, downloads, health checks)."""
        now = self.clock()
        if self._renewed_at is not None and now - self._renewed_at < self.tick_s:
            return
        if not self.backend.renew_coordinator(self.worker_id, self.coordinator_lease_s):
            self._demote()
            raise _Demoted()
        self._renewed_at = now
        self._renew_jobs()

    def _renew_jobs(self) -> None:
        for run in list(self._runs.values()):
            if self._runs.get(run.job_id) is not run:
                continue
            try:
                run.claim = self.backend.renew(run.claim, self.job_lease_s)
            except LeaseLost:
                self._stale(run, "renew", expired=True)

    def _stale(self, run: _Run, operation: str, expired: bool = False) -> None:
        self._drop(run)
        if expired:
            self.obs.emit(ev.QUEUE_LEASE_EXPIRED, job_id=run.job_id, worker=self.worker_id)
        self.obs.inc("gpu_queue_stale_writes_total")
        self.obs.emit(ev.QUEUE_STALE_WRITE_REFUSED, job_id=run.job_id, operation=operation, worker=self.worker_id)

    def _drop(self, run: _Run) -> None:
        if self._runs.get(run.job_id) is run:
            self._runs.pop(run.job_id, None)
        shutil.rmtree(self.workdir / run.job_id, ignore_errors=True)

    def _w(self, run: _Run, operation: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """A fenced write: refused (and never retried) once the lease is lost."""
        if self._runs.get(run.job_id) is not run:
            raise _Dropped()
        try:
            return fn(*args, **kwargs)
        except LeaseLost:
            self._stale(run, operation)
            raise _Dropped() from None

    def _update(self, run: _Run, **fields: Any) -> None:
        job = self._w(run, "update", self.backend.update, run.claim, **fields)
        run.claim = Claim(job=job, token=run.claim.token, lease_until=run.claim.lease_until)

    # ------------------------------------------------------------------
    # recovery
    # ------------------------------------------------------------------

    def recover(self) -> None:
        """Find every job an earlier coordinator left in flight and take it back:
        running on a machine -> adopt, finished there -> deliver, nowhere -> requeue
        MACHINE_LOST. A job whose old lease has not run out is retried every tick."""
        states = {JobState.QUEUED, JobState.CLAIMED, JobState.RUNNING, JobState.DELIVERING}
        try:
            jobs = self.backend.list(states, limit=100_000)
        except Exception as exc:  # noqa: BLE001
            self.obs.emit(ev.PROVIDER_ERROR, operation="recover", error=str(exc)[:500])
            return
        for job in jobs:
            if job.id not in self._runs and (self._unsettled(job) or self._in_flight(job)):
                self._adopt.add(job.id)
        self._retry_adoptions()

    @staticmethod
    def _in_flight(job: QueueJob) -> bool:
        lim = job.limits or {}
        return bool(lim.get("remote_id")) and lim.get("attempt") == len(job.attempts) + 1

    @staticmethod
    def _unsettled(job: QueueJob) -> bool:
        """The attempt was closed but the settle that followed never landed."""
        settling = (job.limits or {}).get("settling")
        return bool(settling) and settling.get("id") != (job.result or {}).get("settle_id")

    def _retry_adoptions(self) -> None:
        for job_id in sorted(self._adopt):
            if job_id in self._runs:
                self._adopt.discard(job_id)
                continue
            claim = self.backend.claim(self.worker_id, self.job_lease_s, job_id=job_id)
            if claim is None:
                job = self.backend.get(job_id)
                if job is None or job.state in FINAL_STATES or not (self._in_flight(job) or self._unsettled(job)):
                    self._adopt.discard(job_id)
                continue
            self._adopt.discard(job_id)
            self._claimed(claim, recovered=True)

    def _claimed(self, claim: Claim, recovered: bool = False) -> Optional[_Run]:
        """Take a fresh claim in hand. Returns the run when it waits for a machine."""
        job = claim.job
        n = len(job.attempts) + 1
        run = _Run(claim=claim, number=n, remote_id=f"{job.id}-a{n}")
        self._runs[job.id] = run
        self._adopt.discard(job.id)
        self._report.claimed += 1
        self.obs.emit(ev.QUEUE_JOB_CLAIMED, job_id=job.id, worker=self.worker_id, attempt=n, recovered=recovered)
        try:
            if self._unsettled(job):
                s = job.limits["settling"]
                self._finish_settle(run, JobState(s["state"]), dict(s["result"]))
                return None
            if self._in_flight(job):
                lim = job.limits or {}
                run.slot = lim.get("slot")
                run.started_at = lim.get("started_at")
                run.limits = _limits_from(lim)
                run.locate = True
                self._locate(run)
                return run if run.phase == "pending" and not run.locate else None
        except _Dropped:
            return None
        return run

    def _locate(self, run: _Run) -> None:
        """Look for the run's remote id on every machine."""
        conclusive = True
        for m in self.pool.machines():
            if m.state in (MachineState.GONE, MachineState.RETIRED, MachineState.STOPPED):
                continue
            if m.state not in _ANSWERING:
                conclusive = False  # booting, unresponsive or unhealthy: cannot tell yet
                continue
            try:
                listed = self.pool.remote_jobs(m)
            except Exception:  # noqa: BLE001
                conclusive = False
                continue
            rec = next((j for j in listed if str(j.get("id")) == run.remote_id), None)
            if rec is None:
                continue
            run.locate = False
            run.slot = m.slot
            run.started_at = run.started_at or rec.get("started_at") or self.clock()
            lim = dict(run.job.limits or {})
            lim["adopted_by"] = self.worker_id
            if rec.get("state") == "running":
                run.phase = "running"
                self._update(run, state=JobState.RUNNING, machine=m.slot, limits=lim)
            else:
                run.phase = "delivering"
                run.exit_code = rec.get("returncode")
                run.remote_state = rec.get("state")
                run.delivering_since = self.clock()
                self._update(run, state=JobState.DELIVERING, machine=m.slot, limits=lim)
            self.obs.emit(
                ev.QUEUE_JOB_STARTED, job_id=run.job_id, attempt=run.number, slot=m.slot,
                remote_id=run.remote_id, adopted=True, remote_state=rec.get("state"),
            )
            return
        if conclusive:
            run.locate = False
            self._fault(run, Outcome.MACHINE_LOST, "not found on any machine after the coordinator changed")

    # ------------------------------------------------------------------
    # step 4: machines that failed, then each running attempt
    # ------------------------------------------------------------------

    def _machine_faults(self) -> None:
        machines = self.pool.machines()
        for m in machines:
            if m.state in _FAULTS:
                outcome, reason, mark_bad = _FAULTS[m.state]
                self._retire(m, outcome, reason, mark_bad)
        tracked = {m.slot: m for m in self.pool.machines()}
        for run in list(self._runs.values()):
            if run.slot is None or run.locate or (run.phase == "delivering" and run.fetched is not None):
                continue
            m = tracked.get(run.slot)
            why = None
            if m is None:
                why = f"the pool no longer has machine {run.slot}"
            elif m.state == MachineState.STOPPED:
                why = f"machine {run.slot} stopped"
            if why is None:
                continue
            try:
                self._fault(run, Outcome.MACHINE_LOST, why)
            except _Dropped:
                pass

    def _retire(self, m: PoolMachine, outcome: Outcome, reason: str, mark_bad: bool) -> None:
        """Retire a machine, then requeue what ran on it. Nothing is requeued while the
        machine may still exist: destroying it is the only proof the job stopped."""
        on = [
            r for r in self._runs.values()
            if r.slot == m.slot and not (r.phase == "delivering" and r.fetched is not None)
        ]
        try:
            result = self.pool.retire(m, reason, mark_bad=mark_bad)
        except Exception as exc:  # noqa: BLE001 - e.g. still booting; next tick
            self.obs.emit(ev.PROVIDER_ERROR, operation="retire", slot=m.slot, error=str(exc)[:500])
            return
        if not result.destroyed:
            return
        for run in on:
            if self._runs.get(run.job_id) is not run:
                continue
            pulled = result.pulled.get(run.remote_id)
            try:
                if run.phase == "delivering" and run.exit_code == 0 and pulled is not None:
                    run.fetched = pulled  # it finished well; deliver from what retire saved
                    continue
                detail = reason
                if run.remote_id in result.lost:
                    detail += "; its outputs could not be saved"
                    self.obs.emit(
                        ev.QUEUE_OUTPUTS_LOST, job_id=run.job_id, attempt=run.number,
                        slot=m.slot, remote_id=run.remote_id, reason=reason,
                    )
                self._fault(run, outcome, detail, outputs_dir=pulled)
            except _Dropped:
                pass

    def _watch(self) -> None:
        tracked = {m.slot: m for m in self.pool.machines()}
        listings: dict[str, Optional[dict[str, dict]]] = {}
        for run in list(self._runs.values()):
            if self._runs.get(run.job_id) is not run or run.phase == "pending":
                continue
            m = tracked.get(run.slot or "")
            try:
                if run.phase == "delivering" and (run.fetched is not None or (m is not None and m.state in _ANSWERING)):
                    self._deliver(run, m)
                    continue
                if m is None or m.state not in _ANSWERING:
                    continue
                if m.slot not in listings:
                    try:
                        listings[m.slot] = {str(j.get("id")): j for j in self.pool.remote_jobs(m)}
                    except Exception:  # noqa: BLE001 - poll marks it unresponsive in time
                        listings[m.slot] = None
                listed = listings[m.slot]
                if listed is None:
                    continue
                rec = listed.get(run.remote_id)
                self._follow(run, m, rec)
            except _Dropped:
                continue
            except _Demoted:
                raise
            except Exception as exc:  # noqa: BLE001 - one job never stops the tick
                self.obs.emit(ev.PROVIDER_ERROR, operation="watch", job_id=run.job_id, error=str(exc)[:500])
            self._heartbeat()

    def _follow(self, run: _Run, m: PoolMachine, rec: Optional[dict]) -> None:
        if rec is None:
            if run.phase == "cancelling":
                self._cancelled(run, None)
            else:
                self._fault(run, Outcome.MACHINE_LOST, f"machine {m.slot} no longer has {run.remote_id}")
            return
        state = rec.get("state")
        if state == "running":
            if run.phase == "running" and run.job.cancel_requested:
                try:
                    self.pool.cancel(m, run.remote_id)
                except Exception as exc:  # noqa: BLE001 - asked again next tick
                    self.obs.emit(ev.PROVIDER_ERROR, operation="cancel", job_id=run.job_id, error=str(exc)[:500])
                    return
                run.phase = "cancelling"
            return
        if run.phase == "cancelling":
            self._cancelled(run, m)
            return
        run.exit_code = rec.get("returncode")
        run.remote_state = state
        run.phase = "delivering"
        run.delivering_since = self.clock()
        self._update(run, state=JobState.DELIVERING)
        self._deliver(run, m)

    def _cancelled(self, run: _Run, m: Optional[PoolMachine]) -> None:
        """The cancelled run stopped: keep what it wrote, settle CANCELLED."""
        if m is not None:
            dest = self._dir(run) / f"out-a{run.number}"
            try:
                shutil.rmtree(dest, ignore_errors=True)
                self.pool.fetch_outputs(m, run.remote_id, dest)
                self._store_attempt(run, dest)
            except _Dropped:
                raise
            except Exception:  # noqa: BLE001 - a cancelled job's outputs are best effort
                pass
        attempt = self._attempt(run, Outcome.CANCELLED, "cancel requested")
        self._settle(run, JobState.CANCELLED, {"outcome": "cancelled", "reason": "cancel requested"}, attempt)

    def _deliver(self, run: _Run, m: Optional[PoolMachine]) -> None:
        """fetch -> health check -> store -> verify -> settle; any delivery problem keeps
        the job DELIVERING and is tried again next tick."""
        if run.fetched is None:
            if m is None:
                return
            dest = self._dir(run) / f"out-a{run.number}"
            try:
                shutil.rmtree(dest, ignore_errors=True)
                self.pool.fetch_outputs(m, run.remote_id, dest)
            except Exception as exc:  # noqa: BLE001
                self._delivery_failed(run, "fetch", [f"{type(exc).__name__}: {str(exc)[:300]}"])
                return
            run.fetched = dest
            self._heartbeat()

        ok_exit = run.exit_code == 0 and run.remote_state in (None, "succeeded")
        if not run.checked:
            run.checked = True
            if m is not None and m.state in _ANSWERING and (not ok_exit or self.queue.health_check.between_jobs):
                health = self.pool.health_check(m)
                self._heartbeat()
                if not health.ok:
                    if not ok_exit and run.remote_state != "timeout":
                        # A crash followed by a failed health check is the machine's fault.
                        self._fault(
                            run, Outcome.UNHEALTHY,
                            f"exit {run.exit_code}, then the health check failed: {health.detail[-300:]}",
                            outputs_dir=run.fetched,
                        )
                        self._retire(m, Outcome.UNHEALTHY, "unhealthy", True)
                        return
                    self._retire(m, Outcome.UNHEALTHY, "unhealthy", True)
                elif run.remote_state == "lost":
                    self._fault(
                        run, Outcome.MACHINE_LOST, "the job's process vanished without an exit code",
                        outputs_dir=run.fetched,
                    )
                    return

        try:
            ref = self.outputs.put_outputs(run.job_id, run.number, run.fetched, success=ok_exit)
            problems = self.outputs.verify(ref)
        except Exception as exc:  # noqa: BLE001
            ref, problems = None, [f"{type(exc).__name__}: {str(exc)[:300]}"]
        if problems or ref is None:
            self._delivery_failed(run, "store", problems)
            return
        self._stored(run, ref)

        if ok_exit:
            outcome, state = Outcome.SUCCEEDED, JobState.SUCCEEDED
            reason = ""
        elif run.remote_state == "timeout":
            outcome, state = Outcome.TIMEOUT, JobState.FAILED
            reason = f"timed out after {run.limits.timeout_s:g}s" if run.limits else "timed out"
        else:
            outcome, state = Outcome.JOB_FAILED, JobState.FAILED
            reason = f"exit {run.exit_code}" if run.exit_code is not None else f"ended {run.remote_state}"
        attempt = self._attempt(run, outcome, reason)
        result = {
            "outcome": outcome.value, "reason": reason, "exit_code": run.exit_code,
            "outputs": ref.location, "files": len(ref.files), "attempts": run.number,
        }
        self._settle(run, state, result, attempt, outputs_ref=asdict(ref))

    def _delivery_failed(self, run: _Run, stage: str, problems: list[str]) -> None:
        self.obs.emit(
            ev.QUEUE_OUTPUTS_DELIVERY_FAILED, job_id=run.job_id, attempt=run.number,
            stage=stage, problems=list(problems)[:10],
        )

    def _stored(self, run: _Run, ref: OutputRef) -> None:
        total = sum(int(f.get("bytes") or 0) for f in ref.files)
        digest = hashlib.sha256(json.dumps(ref.files, sort_keys=True).encode()).hexdigest()
        self.obs.inc("gpu_queue_outputs_bytes_total", total)
        self.obs.emit(
            ev.QUEUE_OUTPUTS_STORED, job_id=run.job_id, attempt=run.number, files=len(ref.files),
            bytes=total, sha256=digest, location=ref.location,
        )

    def _store_attempt(self, run: _Run, src: Path) -> None:
        """Keep a failed attempt's outputs (for the next attempt's $JAMBU_PREVIOUS_OUTPUT_DIR)."""
        if not src.is_dir() or not any(p.is_file() for p in src.rglob("*")):
            return
        try:
            ref = self.outputs.put_outputs(run.job_id, run.number, src, success=False)
            problems = self.outputs.verify(ref)
        except Exception as exc:  # noqa: BLE001
            ref, problems = None, [f"{type(exc).__name__}: {str(exc)[:300]}"]
        if problems or ref is None:
            self._delivery_failed(run, "store_attempt", problems)
            return
        self._stored(run, ref)
        self._update(run, outputs_ref=asdict(ref))

    # ------------------------------------------------------------------
    # closing an attempt
    # ------------------------------------------------------------------

    def _attempt(self, run: _Run, outcome: Outcome, detail: str) -> Attempt:
        now = self.clock()
        return Attempt(
            number=run.number, machine=run.slot, remote_id=run.remote_id,
            started_at=float(run.started_at if run.started_at is not None else now),
            ended_at=now, outcome=outcome, exit_code=run.exit_code, detail=detail[:500],
        )

    def _fault(self, run: _Run, outcome: Outcome, detail: str, outputs_dir: Optional[Path] = None) -> None:
        """The attempt ended for a reason that is not the job's: retry, or give up at max_attempts."""
        if outputs_dir is not None:
            self._store_attempt(run, Path(outputs_dir))
        attempt = self._attempt(run, outcome, detail)
        attempts = run.job.attempts + [attempt]
        if len(attempts) >= self.queue.max_attempts:
            reasons = "; ".join(
                f"attempt {a.number}: {a.outcome.value if a.outcome else '?'}" + (f" ({a.detail})" if a.detail else "")
                for a in attempts
            )
            result = {
                "outcome": outcome.value,
                "reason": f"gave up after {len(attempts)} attempts: {reasons}",
                "attempts": [a.to_dict() for a in attempts],
            }
            self._settle(run, JobState.FAILED, result, attempt)
            self.obs.emit(
                ev.QUEUE_JOB_DEAD, job_id=run.job_id, attempts=len(attempts),
                outcomes=[a.outcome.value if a.outcome else None for a in attempts],
            )
            return
        self._w(run, "requeue", self.backend.requeue, run.claim, attempt)
        self.obs.inc("gpu_queue_attempts_total", outcome=outcome.value)
        self.obs.inc("gpu_queue_retries_total", cause=outcome.value)
        self.obs.emit(
            ev.QUEUE_JOB_RETRIED, job_id=run.job_id, attempt=run.number, cause=outcome.value,
            slot=run.slot, detail=detail[:500],
        )
        self._report.requeued += 1
        self._drop(run)

    def _settle(
        self, run: _Run, state: JobState, result: dict, attempt: Optional[Attempt] = None,
        outputs_ref: Optional[dict] = None,
    ) -> None:
        """Record the closed attempt, then settle. The attempt and the result are written
        first with a settle id, so a crash between the two writes is finished by whoever
        claims the job next instead of running it again."""
        settle_id = secrets.token_hex(8)
        result = {**result, "settle_id": settle_id}
        fields: dict[str, Any] = {
            "limits": {**(run.job.limits or {}), "settling": {"id": settle_id, "state": state.value, "result": result}},
        }
        if attempt is not None:
            fields["attempts"] = run.job.attempts + [attempt]
            self.obs.inc("gpu_queue_attempts_total", outcome=attempt.outcome.value if attempt.outcome else "none")
        if outputs_ref is not None:
            fields["outputs_ref"] = outputs_ref
        self._update(run, **fields)
        self._finish_settle(run, state, result)

    def _finish_settle(self, run: _Run, state: JobState, result: dict) -> None:
        job = self._w(run, "settle", self.backend.settle, run.claim, state, result)
        now = self.clock()
        self._report.settled += 1
        self.obs.inc("gpu_queue_jobs_total", final_state=state.value)
        fields: dict[str, Any] = {"job_id": run.job_id, "attempts": len(job.attempts), "outcome": result.get("outcome")}
        if run.started_at is not None:
            run_s = max(0.0, now - float(run.started_at))
            fields["run_s"] = round(run_s, 3)
            self.obs.timing("gpu_queue_job_run_seconds", run_s)
        cost = self._job_cost(job)
        if cost is not None:
            fields["cost_usd"] = round(cost, 6)
            self.obs.timing("gpu_queue_job_cost_usd", cost)
        if state == JobState.SUCCEEDED:
            self.obs.emit(ev.QUEUE_JOB_SUCCEEDED, outputs=result.get("outputs"), **fields)
        elif state == JobState.FAILED:
            self.obs.emit(ev.QUEUE_JOB_FAILED, reason=str(result.get("reason") or "")[:500], **fields)
        else:
            self.obs.emit(ev.QUEUE_CONTROL, action="cancelled", actor=self.worker_id, **fields)
        self._drop(run)

    def _job_cost(self, job: QueueJob) -> Optional[float]:
        spend = getattr(self.pool, "spend", None)
        if spend is None:
            return None
        try:
            by_job = spend.by_job()
        except Exception:  # noqa: BLE001
            return None
        ids = {a.remote_id for a in job.attempts}
        return float(sum(v for k, v in by_job.items() if k in ids))

    # ------------------------------------------------------------------
    # step 5: claim and dispatch within free capacity
    # ------------------------------------------------------------------

    def _occupied(self, slot: str) -> int:
        return sum(
            1 for r in self._runs.values()
            if r.slot == slot and (
                r.phase in ("pending", "running", "cancelling") or (r.phase == "delivering" and not r.checked)
            )
        )

    def _free(self, m: PoolMachine) -> int:
        if m.state not in _LIVE:
            return 0
        used = self._occupied(m.slot)
        if m.state == MachineState.BUSY:
            used = max(used, 1)  # something runs there that this coordinator may not know
        return max(0, max(1, self.queue.jobs_per_machine) - used)

    def _place(self) -> None:
        paused = self.backend.paused()
        if not paused and self._adopt:
            self._retry_adoptions()
        for run in sorted(
            (r for r in self._runs.values() if r.phase == "pending"),
            key=lambda r: r.job.submitted_at,
        ):
            if self._runs.get(run.job_id) is not run:
                continue
            try:
                if run.locate:
                    self._locate(run)
                    continue
                machines = [m for m in self.pool.machines() if m.state in _LIVE]
                if run.slot is not None:
                    target = next((m for m in machines if m.slot == run.slot), None)
                else:
                    target = next((m for m in machines if self._free(m) > 0), None)
                if target is not None:
                    self._launch(run, target)
            except _Dropped:
                continue
            self._heartbeat()
        if paused:
            return
        for m in self.pool.machines():
            while self._free(m) > 0:
                claim = self.backend.claim(self.worker_id, self.job_lease_s)
                if claim is None:
                    return
                run = self._claimed(claim)
                if run is None:
                    continue
                try:
                    self._launch(run, m)
                except _Dropped:
                    continue
                self._heartbeat()
                if self._runs.get(run.job_id) is run and run.phase == "pending":
                    break  # this machine cannot take work right now

    def _launch(self, run: _Run, m: PoolMachine) -> None:
        job = run.job
        now = self.clock()
        if job.cancel_requested:
            self._settle(run, JobState.CANCELLED, {"outcome": "cancelled", "reason": "cancelled before it started"})
            return
        if run.limits is None:
            ctx = PreflightContext(
                hourly_usd=float(m.hourly_usd or self.pool.next_offer_usd() or 0.0),
                card_vram_gb=self.pool.card_vram_gb(),
                defaults=self.queue.defaults,
                attempt=run.number,
            )
            verdict = self.preflight(job, ctx)
            if not verdict.ok:
                self.obs.emit(ev.QUEUE_PREFLIGHT_REJECTED, job_id=job.id, attempt=run.number, reason=verdict.reason)
                attempt = Attempt(
                    number=run.number, machine=None, remote_id=run.remote_id, started_at=now, ended_at=now,
                    outcome=Outcome.PREFLIGHT_REJECTED, detail=verdict.reason[:500],
                )
                self._settle(
                    run, JobState.FAILED, {"outcome": Outcome.PREFLIGHT_REJECTED.value, "reason": verdict.reason}, attempt
                )
                return
            run.limits = verdict.limits
        limits = run.limits
        until = now + limits.timeout_s + limits.deadline_margin_s
        try:
            self.pool.extend_deadline(m, max(until, m.deadline_at or 0.0))
        except Exception as exc:  # noqa: BLE001 - no deadline, no dispatch; next tick
            self.obs.emit(ev.PROVIDER_ERROR, operation="extend_deadline", slot=m.slot, error=str(exc)[:500])
            return

        work = self._dir(run)
        raw = work / f"bundle-a{run.number}.raw.tar.gz"
        try:
            self.outputs.get_bundle(job.bundle_ref, raw)
        except FileNotFoundError as exc:
            attempt = Attempt(
                number=run.number, machine=None, remote_id=run.remote_id, started_at=now, ended_at=now,
                outcome=Outcome.JOB_FAILED, detail=f"bundle missing: {exc}"[:500],
            )
            self._settle(run, JobState.FAILED, {"outcome": "job_failed", "reason": f"bundle missing: {exc}"}, attempt)
            return
        except Exception as exc:  # noqa: BLE001 - the store may be back next tick
            self.obs.emit(ev.PROVIDER_ERROR, operation="get_bundle", job_id=job.id, error=str(exc)[:500])
            return
        bundle, env, manifest = raw, {}, dict(job.manifest or {})
        if run.number > 1 and job.outputs_ref:
            bundle = self._with_previous(run, raw, work)
            if bundle != raw:
                env["JAMBU_PREVIOUS_OUTPUT_DIR"] = f"{REMOTE_JOBS_DIR}/{run.remote_id}/src/{PREVIOUS_DIR}"
                command = manifest.get("run") or manifest.get("command")
                if isinstance(command, str):
                    manifest["run"] = (
                        f'export JAMBU_PREVIOUS_OUTPUT_DIR="$JAMBU_JOB_DIR/src/{PREVIOUS_DIR}"\n{command}'
                    )
        manifest["timeout_s"] = limits.timeout_s

        if run.started_at is None:
            run.started_at = now
        lim = {
            **asdict(limits), "attempt": run.number, "remote_id": run.remote_id, "slot": m.slot,
            "started_at": run.started_at, "deadline_at": until, "worker": self.worker_id,
        }
        self._update(run, limits=lim, machine=m.slot)
        run.slot = m.slot  # from here a retry goes to this machine only
        try:
            outcome = self.pool.dispatch(m, run.remote_id, bundle, manifest, env)
        except Exception as exc:  # noqa: BLE001 - "already_running" next time if it did start
            self.obs.emit(ev.PROVIDER_ERROR, operation="dispatch", job_id=job.id, slot=m.slot, error=str(exc)[:500])
            return
        run.phase = "running"
        self._update(run, state=JobState.RUNNING)
        self._report.dispatched += 1
        wait_s = max(0.0, now - float(job.submitted_at))
        self.obs.timing("gpu_queue_job_wait_seconds", wait_s)
        self.obs.emit(
            ev.QUEUE_JOB_STARTED, job_id=job.id, attempt=run.number, slot=m.slot, remote_id=run.remote_id,
            dispatch=outcome, wait_s=round(wait_s, 3), timeout_s=limits.timeout_s, deadline_at=until,
            previous_outputs=bool(env),
        )

    def _with_previous(self, run: _Run, raw: Path, work: Path) -> Path:
        """The bundle plus the last attempt's verified outputs under .jambu-previous/;
        the raw bundle when they cannot be had (the job starts from scratch)."""
        prev = work / "previous"
        out = work / f"bundle-a{run.number}.tar.gz"
        try:
            shutil.rmtree(prev, ignore_errors=True)
            self.outputs.fetch_previous(OutputRef(**run.job.outputs_ref), prev)
            with tarfile.open(raw, "r:*") as src, tarfile.open(out, "w:gz") as dst:
                for member in src.getmembers():
                    name = member.name[2:] if member.name.startswith("./") else member.name
                    if name == PREVIOUS_DIR or name.startswith(PREVIOUS_DIR + "/"):
                        continue
                    dst.addfile(member, src.extractfile(member) if member.isfile() else None)
                dst.add(prev, arcname=PREVIOUS_DIR)
        except Exception as exc:  # noqa: BLE001
            self.obs.emit(
                ev.QUEUE_OUTPUTS_LOST, job_id=run.job_id, attempt=run.number,
                reason=f"previous outputs unavailable: {type(exc).__name__}: {str(exc)[:300]}",
            )
            return raw
        return out

    def _dir(self, run: _Run) -> Path:
        d = self.workdir / run.job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # step 6: scale
    # ------------------------------------------------------------------

    def _scale(self) -> None:
        q = self.queue
        now = self.clock()
        paused = self.backend.paused()
        self._stats = stats = self.backend.stats()
        machines = self.pool.machines()
        queued = 0
        if not paused:
            queued = max(0, int(stats.by_state.get(JobState.QUEUED.value, 0)) - len(self._adopt))
            queued += sum(1 for r in self._runs.values() if r.phase == "pending" and r.slot is None and not r.locate)
        views = [MachineView(m.slot, m.state, m.hourly_usd, self._occupied(m.slot)) for m in machines]
        counted = sum(1 for v in views if v.state in ("provisioning", "checking", "ready", "busy"))
        offer = None
        if queued > 0 or counted < q.min_machines:
            try:
                offer = self.pool.next_offer_usd()
            except Exception as exc:  # noqa: BLE001
                self.obs.emit(ev.PROVIDER_ERROR, operation="next_offer", error=str(exc)[:500])
        oldest = stats.oldest_queued_age_s if queued else None
        if oldest is None and queued:
            oldest = 0.0
        inp = ScaleInput(
            now=now, queued=queued, oldest_queued_age_s=oldest, machines=views,
            pool_usd_per_hour=pool_usd_per_hour(machines), next_offer_usd=offer,
            min_machines=q.min_machines, max_machines=q.max_machines, max_usd_per_hour=q.max_usd_per_hour,
            scale_up_after_s=q.scale_up_after_s, jobs_per_machine=q.jobs_per_machine,
            last_open_at=self.pool.last_failed_at, open_backoff_s=self.pool.open_backoff_s,
        )
        decision = decide(inp)
        key = (decision.open, decision.reason, decision.blocked_by)
        if key != self._last_decision:
            self._last_decision = key
            self.obs.emit(
                ev.POOL_SCALE_DECISION, open=decision.open, reason=decision.reason, blocked_by=decision.blocked_by,
                queued=queued, oldest_queued_age_s=oldest, machines=counted,
                pool_usd_per_hour=inp.pool_usd_per_hour, next_offer_usd=offer, paused=paused,
            )
        if decision.blocked_by != self._last_blocked:
            self._last_blocked = decision.blocked_by
            if decision.blocked_by in ("max_machines", "max_usd_per_hour"):
                self.obs.emit(
                    ev.POOL_CAP_BLOCKED, cap=decision.blocked_by, queued=queued,
                    pool_usd_per_hour=inp.pool_usd_per_hour, next_offer_usd=offer,
                )
        for _ in range(decision.open):
            try:
                opened = self.pool.open()
            except Exception as exc:  # noqa: BLE001
                self.obs.emit(ev.PROVIDER_ERROR, operation="pool_open", error=str(exc)[:500])
                break
            if opened is None:
                break  # backing off or at max_machines: not now
            self._report.opened += 1
        by_slot = {m.slot: m for m in machines}
        for slot in decision.keepalive:
            m = by_slot.get(slot)
            if m is None:
                continue
            try:
                self.pool.keepalive(m)
            except Exception as exc:  # noqa: BLE001
                self.obs.emit(ev.PROVIDER_ERROR, operation="keepalive", slot=slot, error=str(exc)[:500])

    # ------------------------------------------------------------------
    # step 8: metrics
    # ------------------------------------------------------------------

    def _metrics(self) -> None:
        now = self.clock()
        try:
            if self.leader and self._renewed_at is not None:
                self.obs.gauge("gpu_queue_coordinator_heartbeat_age_seconds", max(0.0, now - self._renewed_at))
            self.obs.gauge("gpu_queue_coordinator_leader", 1.0 if self.leader else 0.0)
            if self.leader:
                self.obs.gauge("gpu_queue_paused", 1.0 if self.backend.paused() else 0.0)
                stats = self._stats or self.backend.stats()
                self.obs.gauge("gpu_queue_depth", float(stats.by_state.get(JobState.QUEUED.value, 0)))
                self.obs.gauge("gpu_queue_oldest_queued_seconds", float(stats.oldest_queued_age_s or 0.0))
                pending = [r for r in self._runs.values() if r.phase == "delivering"]
                self.obs.gauge("gpu_queue_outputs_pending", float(len(pending)))
                oldest = max((now - (r.delivering_since or now) for r in pending), default=0.0)
                self.obs.gauge("gpu_queue_outputs_oldest_pending_seconds", oldest)
            self._stats = None
            self.obs.write()
        except Exception as exc:  # noqa: BLE001 - metrics never stop the loop
            self.obs.emit(ev.PROVIDER_ERROR, operation="metrics", error=str(exc)[:500])
