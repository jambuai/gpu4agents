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
7. the auditor (``audit.py``) every 12 ticks, and once on ``recover()``;
8. metrics.

Leases do not depend on the tick: ``run()`` starts a heartbeat thread that renews
the coordinator lease and every job lease every lease/3, so an upload, a download
or a health check that outlasts a tick never lets a lease run out. Backend calls
from the tick and the heartbeat are serialized by one lock. When the heartbeat
cannot renew the coordinator lease, ``leader`` drops at once, and nothing is
claimed or dispatched without it (``_check_lead`` before each claim and dispatch).

What is in flight is in the queue, not only in this process: before a dispatch the
job's ``limits`` record the attempt number, remote id, slot, instance id, owner and
the machine's deadline. A coordinator that finds such a job (after a crash, a
takeover or an expired lease) looks for that remote id and adopts the run instead
of starting it twice. A run on a machine this coordinator cannot see (another
host's pool) is requeued only after that machine's deadline plus a grace, when its
watchdog has ended it.

Writes are fenced by the claim token: a ``LeaseLost`` drops the job from memory, is
reported as a stale write, and the write is never retried. Local files (fetched
outputs) are kept until they are stored and verified or the job is settled.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
from ..core.job import UPLOAD_TIMEOUT_S
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
MIN_COORDINATOR_LEASE_S = 15.0
JOB_LEASE_TICKS = 12
MIN_JOB_LEASE_S = 60.0
# A machine this coordinator cannot see is trusted to be gone this long after its deadline.
DEADLINE_GRACE_S = 300.0
# A machine that answers without the remote id may still be receiving its upload.
UPLOAD_SLACK_S = 60.0
MAX_DISPATCH_FAILURES = 3
MAX_FETCH_FAILURES = 5  # machine-side delivery failures (fetch, health check) before retiring it
STORE_BACKOFF_MAX_S = 300.0
# An in-flight record with no deadline at all (should not happen): the longest it may wait.
NO_DEADLINE_CAP_S = 24 * 3600.0
REMOTE_ID_MAX = 64  # the watchdog's job runner: [A-Za-z0-9][A-Za-z0-9_.-]{0,63}
PREVIOUS_DIR = ".jambu-previous"
_OUT = re.compile(r"out-a([0-9]+)")
# The watchdog's job runner default (JAMBU_JOBS_DIR); the command also exports the
# real path, so a machine with another jobs directory still gets it right.
REMOTE_JOBS_DIR = "/workspace/jobs"

_LIVE = (MachineState.READY, MachineState.BUSY)
_ANSWERING = (MachineState.READY, MachineState.BUSY, MachineState.DRAINING, MachineState.HELD)
_DEAD = (MachineState.GONE, MachineState.RETIRED, MachineState.STOPPED)
_FAULTS = {
    MachineState.GONE: (Outcome.MACHINE_LOST, "machine gone", False),
    MachineState.UNRESPONSIVE: (Outcome.UNRESPONSIVE, "unresponsive", True),
    MachineState.UNHEALTHY: (Outcome.UNHEALTHY, "unhealthy", True),
}


def find_run(
    machines: list[PoolMachine], lim: dict, remote_id: str, list_jobs: Callable[[PoolMachine], list[dict]],
    retired: Any = (),
) -> tuple[Optional[str], Optional[PoolMachine], Optional[dict]]:
    """Look for ``remote_id`` on every machine. Returns (home, machine, record): the
    machine and its record when found; else what is known of the machine the in-flight
    record names: None (not one of these), "dead", "unknown" (cannot tell yet) or
    "answering" (it lists its jobs)."""
    instance, slot = lim.get("instance_id"), lim.get("slot")
    home: Optional[str] = "dead" if instance and instance in retired else None
    for m in machines:
        ours = (m.instance_id == instance) if instance else (m.slot == slot)
        if m.state in _DEAD:
            if ours:
                home = "dead"
            continue
        if m.state not in _ANSWERING:
            if ours:
                home = "unknown"
            continue
        try:
            listed = list_jobs(m)
        except Exception:  # noqa: BLE001
            if ours:
                home = "unknown"
            continue
        if ours:
            home = "answering"
        rec = next((j for j in listed if str(j.get("id")) == remote_id), None)
        if rec is not None:
            return home, m, rec
    return home, None, None


def lost_detail(
    home: Optional[str], started_at: Optional[float], now: float, wait_until: Callable[[], Optional[float]],
) -> Optional[str]:
    """Why a run that ``find_run`` did not find surely runs nowhere, or None while that
    is not certain: its machine cannot be told yet, a machine this pool cannot see has
    not reached its deadline plus a grace, or an upload may still be on its way."""
    if home == "unknown":
        return None
    if home is None:
        # Another host's machine: its watchdog ends the run by its deadline.
        until = wait_until()
        if until is None or now < until:
            return None
        return "previous coordinator's machine unreachable; waited for its deadline"
    if home == "answering" and now < float(started_at or now) + UPLOAD_TIMEOUT_S + UPLOAD_SLACK_S:
        return None  # the upload may still be on its way
    return "not found on any machine after the coordinator changed"


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
    """This coordinator is not the leader (any more): stop placing work."""


class _SkipPlacement(Exception):
    """Placement cannot go on this tick (preflight inputs unavailable)."""


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
    dispatch_failures: int = 0
    fetch_failures: int = 0  # machine-side: fetch, health check
    store_failures: int = 0  # store-side: never the machine's fault
    next_store_at: Optional[float] = None
    last_store_problem: Optional[str] = None
    last_fetch_problem: Optional[str] = None

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
        self.coordinator_lease_s = max(COORDINATOR_LEASE_TICKS * self.tick_s, MIN_COORDINATOR_LEASE_S)
        self.job_lease_s = max(JOB_LEASE_TICKS * self.tick_s, MIN_JOB_LEASE_S)
        self.heartbeat_s = min(self.coordinator_lease_s, self.job_lease_s) / 3
        self.default_margin_s = parse_duration(self.queue.defaults.deadline_margin) or 600.0
        self.workdir = Path(config.state_dir) / "queue" / "work"

        self.leader = False
        self._lock = threading.RLock()  # every backend call: tick and heartbeat thread
        self._runs: dict[str, _Run] = {}
        self._adopt: set[str] = set()  # in-flight jobs whose old lease has not run out yet
        self._retired_instances: set[str] = set()
        self._first_seen: dict[str, float] = {}  # job id -> when a deadline-less record was first met
        self._violations: set[str] = set()
        self._leftover: dict[str, tuple[int, float]] = {}  # "<job>/<n>" -> (failures, next try)
        self._renewed_at: Optional[float] = None
        self._ticks = 0
        self._last_decision: Optional[tuple] = None
        self._last_blocked: Optional[str] = None
        self._errors: dict[str, str] = {}
        self._stats = None
        self._report = TickReport(0, 0, 0, 0, 0, 0)
        # The auditor: called every AUDIT_EVERY_TICKS ticks and on recover(), returns violations.
        from .audit import AuditMemory, audit

        self.audit_memory = AuditMemory()
        self._audit: Optional[Callable[[], Any]] = lambda: audit(
            self.backend, self.pool, self.outputs, self.obs, repair=True, coordinator=self,
        )

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def config_hash(self) -> str:
        body = json.dumps(self.queue.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def run(self, stop: threading.Event) -> None:
        """Tick every ``queue.tick`` until ``stop`` is set; standby while another
        coordinator holds the lease. A heartbeat thread keeps the leases alive in
        between. Releases the lease on the way out."""
        self.obs.emit(
            ev.COORDINATOR_STARTED, worker=self.worker_id, config_hash=self.config_hash(),
            backend=getattr(self.backend, "name", None), tick_s=self.tick_s,
        )
        beats = threading.Event()

        def heartbeat() -> None:
            while not beats.wait(self.heartbeat_s):
                try:
                    self.beat()
                except Exception as exc:  # noqa: BLE001 - the next beat tries again
                    self._error_once("heartbeat", exc)

        thread = threading.Thread(target=heartbeat, name="coordinator-heartbeat", daemon=True)
        thread.start()
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
            beats.set()
            thread.join(5)
            with self._lock:
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
            self._heartbeat()
            with self._lock:
                self._renew_jobs()
            draining = self._drain_requested()
            self._machine_faults()
            self._watch()
            self._store_leftovers()
            self._place()
            if not draining:
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

    def _snapshot(self) -> list[_Run]:
        """The runs in hand, copied under the lock (the heartbeat thread may drop one)."""
        with self._lock:
            return list(self._runs.values())

    def _renew_coordinator(self) -> bool:
        """Renew the coordinator lease. A backend error counts as a refusal once the
        lease window since the last renewal has passed (someone else may hold it now);
        before that it is raised (the tick fails) or reported (the heartbeat)."""
        try:
            return bool(self.backend.renew_coordinator(self.worker_id, self.coordinator_lease_s))
        except Exception:
            if self._renewed_at is None or self.clock() - self._renewed_at >= self.coordinator_lease_s:
                return False
            raise

    def _error_once(self, operation: str, exc: BaseException, **fields: Any) -> None:
        """provider_error once per change of what went wrong, not every tick."""
        message = f"{type(exc).__name__}: {str(exc)[:400]}"
        if self._errors.get(operation) == message:
            return
        self._errors[operation] = message
        self.obs.emit(ev.PROVIDER_ERROR, operation=operation, error=message, **fields)

    def _ok(self, operation: str) -> None:
        self._errors.pop(operation, None)

    # ------------------------------------------------------------------
    # leases
    # ------------------------------------------------------------------

    def _lead(self) -> bool:
        with self._lock:
            if self.leader:
                if self._renew_coordinator():
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
        with self._lock:
            if not self.leader:
                return
            self.leader = False
            dropped = sorted(self._runs)
            # Whoever holds the lease now adopts these; nothing is written for them from
            # here. Their local files stay (an adopter on this host reuses them).
            self._runs.clear()
            self._adopt.clear()
        self.obs.inc("gpu_queue_lease_renew_failures_total")
        self.obs.emit(ev.COORDINATOR_LEASE_LOST, worker=self.worker_id, jobs_dropped=dropped)

    def beat(self) -> None:
        """One heartbeat: renew the coordinator lease and every job lease. ``run()``
        calls it every lease/3 from its own thread; the tick calls it when it has
        been a while. A refused renewal clears ``leader`` at once."""
        with self._lock:
            if not self.leader:
                return
            try:
                renewed = self._renew_coordinator()
            except Exception as exc:  # noqa: BLE001 - still inside the lease window
                self._error_once("renew_coordinator", exc)
                return
            if not renewed:
                self._demote()
                return
            self._ok("renew_coordinator")
            self._renewed_at = self.clock()
            self._renew_jobs()

    def _heartbeat(self) -> None:
        """Beat when the last renewal is older than the heartbeat period."""
        if self._renewed_at is None or self.clock() - self._renewed_at >= self.heartbeat_s:
            self.beat()
        if not self.leader:
            raise _Demoted()

    _check_lead = _heartbeat  # before every claim and every dispatch

    def _renew_jobs(self) -> None:
        with self._lock:
            for run in list(self._runs.values()):
                if self._runs.get(run.job_id) is not run:
                    continue
                try:
                    run.claim = self.backend.renew(run.claim, self.job_lease_s)
                except LeaseLost:
                    self._stale(run, "renew", expired=True)

    def _stale(self, run: _Run, operation: str, expired: bool = False) -> None:
        self._drop(run)
        if self._in_flight(run.job):
            self._adopt.add(run.job_id)  # take it back if its new holder lets it lapse
        if expired:
            self.obs.emit(ev.QUEUE_LEASE_EXPIRED, job_id=run.job_id, worker=self.worker_id)
        self.obs.inc("gpu_queue_stale_writes_total")
        self.obs.emit(ev.QUEUE_STALE_WRITE_REFUSED, job_id=run.job_id, operation=operation, worker=self.worker_id)

    def _drop(self, run: _Run) -> None:
        """Out of memory; local files stay until stored or settled (``_cleanup``)."""
        with self._lock:
            if self._runs.get(run.job_id) is run:
                self._runs.pop(run.job_id, None)

    def _cleanup(self, run: _Run) -> None:
        self._cleanup_job(run.job_id)

    def _cleanup_job(self, job_id: str) -> None:
        """Remove what the job's work dir holds, except every out-a<n> not marked stored
        (with its marker file); the dir itself goes once nothing unstored is left."""
        base = self.workdir / job_id
        if not base.is_dir():
            return
        keep: set[str] = set()
        for child in base.iterdir():
            if child.is_dir() and _OUT.fullmatch(child.name) and not (base / f"{child.name}.stored").exists():
                keep |= {child.name, f"{child.name}.json"}
        for child in list(base.iterdir()):
            if child.name in keep:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
        if not keep:
            shutil.rmtree(base, ignore_errors=True)

    def _store_leftovers(self, only: Optional[str] = None) -> None:
        """Store, as failed attempts, outputs fetched here for attempts that are closed
        but whose store failed (the store was down). Retried with backoff; once stored
        they are marked and, for a job no longer in hand, cleaned up."""
        if not self.workdir.is_dir():
            return
        now = self.clock()
        bases = [self.workdir / only] if only else sorted(p for p in self.workdir.iterdir() if p.is_dir())
        for base in bases:
            if not base.is_dir():
                continue
            job_id = base.name
            pending = sorted(
                (c for c in base.iterdir()
                 if c.is_dir() and _OUT.fullmatch(c.name) and not (base / f"{c.name}.stored").exists()),
                key=lambda c: int(_OUT.fullmatch(c.name).group(1)),
            )
            if not pending:
                continue
            with self._lock:
                held = self._runs.get(job_id)
                job = self.backend.get(job_id)
            if job is None:
                continue
            stored_any = False
            for out in pending:
                k = int(_OUT.fullmatch(out.name).group(1))
                if k > len(job.attempts) or (held is not None and k >= held.number):
                    continue  # still in flight: its own delivery stores it
                key = f"{job_id}/{k}"
                tries, next_at = self._leftover.get(key, (0, 0.0))
                if now < next_at:
                    continue
                try:
                    ref = self.outputs.put_outputs(job_id, k, out, success=False)
                    problems = self.outputs.verify(ref)
                except Exception as exc:  # noqa: BLE001
                    ref, problems = None, [f"{type(exc).__name__}: {str(exc)[:300]}"]
                if problems or ref is None:
                    self._leftover[key] = (tries + 1, now + min(self.tick_s * 2 ** tries, STORE_BACKOFF_MAX_S))
                    self._error_once(f"store_leftover:{key}", RuntimeError("; ".join(problems)[:400]), job_id=job_id)
                    continue
                self._leftover.pop(key, None)
                self._ok(f"store_leftover:{key}")
                self._stored_event(job_id, k, ref)
                self._mark_stored(job_id, out)
                stored_any = True
            if stored_any and held is None:
                self._cleanup_job(job_id)

    def _w(self, run: _Run, operation: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """A fenced write: refused (and never retried) once the lease is lost."""
        with self._lock:
            if self._runs.get(run.job_id) is not run:
                raise _Dropped()
            try:
                return fn(*args, **kwargs)
            except LeaseLost:
                self._stale(run, operation)
                raise _Dropped() from None

    def _update(self, run: _Run, **fields: Any) -> None:
        with self._lock:
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
            with self._lock:
                jobs = self.backend.list(states, limit=100_000)
        except Exception as exc:  # noqa: BLE001
            self.obs.emit(ev.PROVIDER_ERROR, operation="recover", error=str(exc)[:500])
            return
        for job in jobs:
            if job.id not in self._runs and (self._unsettled(job) or self._in_flight(job)):
                self._adopt.add(job.id)
        if self._audit is not None:
            try:
                self._audit()
            except Exception as exc:  # noqa: BLE001 - recovery goes on without it
                self.obs.emit(ev.PROVIDER_ERROR, operation="audit", error=str(exc)[:500])
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

    def _retry_adoptions(self, only: Optional[str] = None) -> None:
        """Claim each remembered in-flight job by id (allowed while paused: re-adoption
        is not new work)."""
        with self._lock:
            ids = sorted(self._adopt) if only is None else [only] if only in self._adopt else []
        for job_id in ids:
            if not self.leader:
                return
            if job_id in self._runs:
                self._adopt.discard(job_id)
                continue
            with self._lock:
                if not self.leader:
                    return
                claim = self.backend.claim(self.worker_id, self.job_lease_s, job_id=job_id)
                job = self.backend.get(job_id) if claim is None else None
            if claim is None:
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
        with self._lock:
            self._runs[job.id] = run
            self._adopt.discard(job.id)
        self._report.claimed += 1
        self.obs.emit(ev.QUEUE_JOB_CLAIMED, job_id=job.id, worker=self.worker_id, attempt=n, recovered=recovered)
        try:
            if self._unsettled(job):
                s = job.limits["settling"]
                state = JobState(s["state"])
                if state != JobState.SUCCEEDED or not self._outputs_bad(job):
                    self._finish_settle(run, state, dict(s["result"]))
                    return None
                self._reopen(run)
                job = run.job
            if self._in_flight(job):
                lim = job.limits or {}
                run.slot = lim.get("slot")
                run.started_at = lim.get("started_at")
                run.limits = _limits_from(lim)
                if self._reuse_fetched(run):
                    self.obs.emit(
                        ev.QUEUE_JOB_STARTED, job_id=job.id, attempt=n, slot=run.slot,
                        remote_id=run.remote_id, adopted=True, remote_state=run.remote_state, local_outputs=True,
                    )
                    self._update(run, state=JobState.DELIVERING)
                    return None
                run.locate = True
                self._locate(run)
                return run if run.phase == "pending" and not run.locate else None
            if len(run.remote_id) > REMOTE_ID_MAX:
                reason = (
                    f"job id too long: the remote id {run.remote_id!r} has {len(run.remote_id)} characters, "
                    f"a machine's job runner takes at most {REMOTE_ID_MAX}"
                )
                self._reject(run, reason)
                return None
        except _Dropped:
            return None
        return run

    def _outputs_bad(self, job: QueueJob) -> bool:
        """The recorded outputs fail verification now. A store that cannot be reached says
        nothing: they were verified before the settle was recorded."""
        if not job.outputs_ref:
            return True
        try:
            return bool(self.outputs.verify(OutputRef(**job.outputs_ref)))
        except Exception:  # noqa: BLE001
            return False

    def _reopen(self, run: _Run) -> None:
        """A settle toward success never landed and its outputs no longer verify: undo the
        unfinished settle (the closed attempt and the outputs it recorded) and deliver the
        attempt again, from this host's copy or from the machine; a run found nowhere is
        requeued like any lost run."""
        job = run.job
        lim = dict(job.limits or {})
        lim.pop("settling", None)
        n = int(lim.get("attempt") or len(job.attempts) or 1)
        attempts = [a for a in job.attempts if a.number != n]
        self._update(run, state=JobState.DELIVERING, limits=lim, attempts=attempts, outputs_ref=None)
        run.number, run.remote_id = n, f"{job.id}-a{n}"
        try:
            (self.workdir / job.id / f"out-a{n}.stored").unlink()
        except OSError:
            pass
        self.obs.emit(
            ev.QUEUE_OUTPUTS_DELIVERY_FAILED, job_id=job.id, attempt=n, stage="verify",
            problems=["the recorded outputs fail verification; delivering again"],
        )

    def _marker(self, run: _Run) -> Path:
        return self.workdir / run.job_id / f"out-a{run.number}.json"

    def _write_marker(self, run: _Run) -> None:
        try:
            self._marker(run).write_text(json.dumps({
                "remote_id": run.remote_id, "exit_code": run.exit_code,
                "remote_state": run.remote_state, "checked": run.checked, "slot": run.slot,
            }))
        except OSError:
            pass

    def _reuse_fetched(self, run: _Run) -> bool:
        """Outputs this host already downloaded for the attempt (a crash or a lost lease
        after the fetch): deliver from them instead of fetching again."""
        try:
            data = json.loads(self._marker(run).read_text())
        except (OSError, ValueError):
            return False
        out = self.workdir / run.job_id / f"out-a{run.number}"
        if data.get("remote_id") != run.remote_id or not out.is_dir():
            return False
        # The machine is the one with the recorded instance id, never a slot that merely
        # has the same name (another pool's s1, or a reused slot).
        instance = (run.job.limits or {}).get("instance_id")
        home = next((m for m in self.pool.machines() if instance and m.instance_id == instance), None)
        run.slot = home.slot if home is not None else None
        run.fetched = out
        run.exit_code = data.get("exit_code")
        run.remote_state = data.get("remote_state")
        run.checked = bool(data.get("checked"))
        run.phase = "delivering"
        run.delivering_since = self.clock()
        return True

    def _locate(self, run: _Run) -> None:
        """Look for the run's remote id; adopt it, or requeue it once it is certain
        that it does not run anywhere."""
        lim = run.job.limits or {}
        now = self.clock()
        home, m, rec = find_run(
            self.pool.machines(), lim, run.remote_id, self.pool.remote_jobs, self._retired_instances,
        )
        if m is not None:
            self._adopt_run(run, m, rec)
            return
        detail = lost_detail(home, run.started_at, now, lambda: self._no_later_than(run.job_id, lim, now))
        if detail is None:
            return
        run.locate = False
        self._fault(run, Outcome.MACHINE_LOST, detail)

    def _no_later_than(self, job_id: str, lim: dict, now: float) -> float:
        """When a run on a machine this coordinator cannot see has surely ended: its
        deadline plus a grace; else started_at + timeout + margin plus the grace; else
        (a record that breaks the contract) a hard cap from when it was first met."""
        def num(key: str) -> Optional[float]:
            try:
                value = lim.get(key)
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        deadline = num("deadline_at")
        if deadline is not None:
            return deadline + DEADLINE_GRACE_S
        started, timeout = num("started_at"), num("timeout_s")
        if started is not None and timeout is not None:
            margin = num("deadline_margin_s")
            return started + timeout + (self.default_margin_s if margin is None else margin) + DEADLINE_GRACE_S
        if job_id not in self._violations:
            self._violations.add(job_id)
            self.obs.inc("gpu_queue_invariant_violations_total", code="inflight_without_deadline")
            self.obs.emit(
                ev.QUEUE_INVARIANT_VIOLATED, code="inflight_without_deadline", job_id=job_id,
                slot=lim.get("slot"), detail="an in-flight record carries no deadline; waiting for the hard cap",
            )
        cap = self.queue.max_machine_lifetime_s or NO_DEADLINE_CAP_S
        since = started if started is not None else self._first_seen.setdefault(job_id, now)
        return since + cap

    def _adopt_run(self, run: _Run, m: PoolMachine, rec: dict) -> None:
        run.locate = False
        run.slot = m.slot
        run.started_at = run.started_at or rec.get("started_at") or self.clock()
        lim = dict(run.job.limits or {})
        lim["adopted_by"] = self.worker_id
        lim["owner"] = self.worker_id
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

    # ------------------------------------------------------------------
    # repairs the auditor asks for (only the claim holder can change a job)
    # ------------------------------------------------------------------

    def whereabouts(
        self, job: QueueJob, list_jobs: Optional[Callable[[PoolMachine], list[dict]]] = None,
    ) -> tuple[Optional[str], Optional[PoolMachine], Optional[dict]]:
        """Where an in-flight job's run is: (None, machine, record) when a machine lists
        it; else (why it surely runs nowhere, None, None) by the same rules as adoption
        (``find_run``, ``lost_detail``), or (None, None, None) while it may still run.
        ``list_jobs`` lets a caller list each machine once (default ``pool.remote_jobs``)."""
        lim = job.limits or {}
        now = self.clock()
        home, m, rec = find_run(
            self.pool.machines(), lim, str(lim.get("remote_id") or ""), list_jobs or self.pool.remote_jobs,
            self._retired_instances,
        )
        if m is not None:
            return None, m, rec
        detail = lost_detail(home, lim.get("started_at"), now, lambda: self._no_later_than(job.id, lim, now))
        return detail, None, None

    def lost(self, job: QueueJob, list_jobs: Optional[Callable[[PoolMachine], list[dict]]] = None) -> Optional[str]:
        """Why an in-flight job surely runs on no machine; None while it may still run."""
        return self.whereabouts(job, list_jobs)[0]

    def requeue_lost(self, job_id: str, list_jobs: Optional[Callable[[PoolMachine], list[dict]]] = None) -> bool:
        """Requeue MACHINE_LOST a running job in hand that surely runs nowhere. False when
        the job is not in hand (held elsewhere, or waiting to be adopted, where the same
        rules apply) or may still run."""
        with self._lock:
            run = self._runs.get(job_id)
        if run is None or not self.leader or run.locate or run.phase not in ("running", "cancelling"):
            return False
        detail = self.lost(run.job, list_jobs)
        if detail is None:
            return False
        try:
            self._fault(run, Outcome.MACHINE_LOST, f"auditor: {detail}")
        except _Dropped:
            return False
        return True

    def cancel_overdue(self, job_id: str, list_jobs: Optional[Callable[[PoolMachine], list[dict]]] = None) -> bool:
        """Cancel a job in hand still running past its deadline and requeue it TIMEOUT,
        keeping what it wrote. Only a run its machine lists as running is touched (a
        finished one is left to delivery); nothing is requeued unless the cancel is taken."""
        with self._lock:
            run = self._runs.get(job_id)
        if run is None or not self.leader or run.phase != "running":
            return False
        _, m, rec = self.whereabouts(run.job, list_jobs)
        if m is None or rec is None or rec.get("state") != "running" or m.slot != run.slot:
            return False
        try:
            self._check_lead()
        except _Demoted:
            return False
        try:
            self.pool.cancel(m, run.remote_id)
        except Exception as exc:  # noqa: BLE001 - the next audit asks again
            self.obs.emit(ev.PROVIDER_ERROR, operation="cancel", job_id=job_id, error=str(exc)[:500])
            return False
        dest: Optional[Path] = self._dir(run) / f"out-a{run.number}"
        try:
            shutil.rmtree(dest, ignore_errors=True)
            self.pool.fetch_outputs(m, run.remote_id, dest)
        except Exception:  # noqa: BLE001 - a timed-out run's outputs are best effort
            dest = None
        try:
            self._fault(run, Outcome.TIMEOUT, "auditor: still running past its deadline; cancelled", outputs_dir=dest)
        except _Dropped:
            return False
        return True

    def redeliver(self, job_id: str) -> bool:
        """A job whose settle toward success never landed and whose outputs fail
        verification: take it back and deliver it again (``_reopen``). True once it no
        longer claims a success it cannot show."""
        if not self.leader or job_id in self._runs:
            return False
        with self._lock:
            self._adopt.add(job_id)
        self._retry_adoptions(only=job_id)
        with self._lock:
            job = self.backend.get(job_id)
        return job is not None and job.state != JobState.SUCCEEDED and not self._unsettled(job)

    # ------------------------------------------------------------------
    # drain now
    # ------------------------------------------------------------------

    def _drain_requested(self) -> bool:
        """Act on a drain request left in the backend (``gpu queue drain --now``). The request
        stays until no machine is left, so a machine still booting is retired on a later tick.
        True while a drain is pending: nothing new is opened meanwhile."""
        try:
            with self._lock:
                wanted = self.backend.drain_requested()
        except Exception as exc:  # noqa: BLE001
            self._error_once("drain_requested", exc)
            return False
        if not wanted:
            return False
        result = self.drain_now("drain-request")
        if result.get("performed") and not result["remaining"] and not result.get("waiting_for") and not self._runs:
            try:
                with self._lock:
                    self.backend.clear_drain()
            except Exception as exc:  # noqa: BLE001 - tried again next tick
                self._error_once("clear_drain", exc)
            return False
        return True

    def drain_now(self, actor: str) -> dict:
        """Pause the queue, retire every machine (retire pulls the outputs) and send the jobs
        that ran there back to the queue; attempts recorded as ``drained`` do not count toward
        ``max_attempts``. A run whose machine was not destroyed stays in hand, and a job this
        coordinator has not adopted yet (its old lease is still alive) leaves its machine alone,
        so nothing is requeued while it may still run and its outputs are not thrown away: the
        drain request stays and is retried. Takes the coordinator lease first when it does not
        hold it; ``performed`` is False when another coordinator does."""
        if not self.leader and not self._lead():
            return {"performed": False, "reason": "another coordinator holds the lease"}
        with self._lock:
            if not self.backend.paused():  # a pending drain pauses once, under the real actor
                self.backend.set_paused(True, actor)
        before = self._report.requeued
        self._retry_adoptions()
        waiting = sorted(self._adopt)
        if waiting:
            remaining = [m.slot for m in self.pool.machines() if m.state not in _DEAD]
            self.obs.emit(ev.QUEUE_CONTROL, action="drain_now_waiting", actor=actor, waiting_for=waiting, remaining=remaining)
            return {"performed": True, "requeued": 0, "remaining": remaining, "waiting_for": waiting}
        for m in self.pool.machines():
            if m.state not in _DEAD:
                self._retire(m, Outcome.DRAINED, "drain now", False)
        live = {m.slot for m in self.pool.machines() if m.state not in _DEAD}
        for run in self._snapshot():
            if self._runs.get(run.job_id) is not run or run.fetched is not None or run.phase == "delivering":
                continue
            if run.slot is not None and run.slot in live:
                continue  # its machine is still there: the next pass retires it first
            try:
                self._fault(run, Outcome.DRAINED, "drain now")
            except _Dropped:
                pass
        remaining = [m.slot for m in self.pool.machines() if m.state not in _DEAD]
        requeued = self._report.requeued - before
        self.obs.emit(ev.QUEUE_CONTROL, action="drain_now_done", actor=actor, requeued=requeued, remaining=remaining)
        return {"performed": True, "requeued": requeued, "remaining": remaining}

    def release(self) -> None:
        """Give up the coordinator lease (a one-shot caller that took it)."""
        with self._lock:
            if self.leader:
                try:
                    self.backend.release_coordinator(self.worker_id)
                except Exception:  # noqa: BLE001 - the lease runs out by itself
                    pass
                self.leader = False

    # ------------------------------------------------------------------
    # step 4: machines that failed, then each running attempt
    # ------------------------------------------------------------------

    def _machine_faults(self) -> None:
        for m in self.pool.machines():
            if m.state in _FAULTS:
                outcome, reason, mark_bad = _FAULTS[m.state]
                self._retire(m, outcome, reason, mark_bad)
        tracked = {m.slot: m for m in self.pool.machines()}
        for run in self._snapshot():
            if run.slot is None or run.locate or run.fetched is not None:
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

    def _retire(
        self, m: PoolMachine, outcome: Outcome, reason: str, mark_bad: bool,
        details: Optional[dict[str, tuple[Outcome, str]]] = None,
    ) -> None:
        """Retire a machine, then requeue what ran on it. Nothing is requeued while the
        machine may still exist: destroying it is the only proof the job stopped.
        ``details`` gives one run its own outcome and reason."""
        details = details or {}
        # A run whose outputs are already here does not need the machine any more.
        on = [r for r in self._snapshot() if r.slot == m.slot and r.fetched is None]
        try:
            result = self.pool.retire(m, reason, mark_bad=mark_bad)
        except Exception as exc:  # noqa: BLE001 - e.g. still booting; next tick
            self.obs.emit(ev.PROVIDER_ERROR, operation="retire", slot=m.slot, error=str(exc)[:500])
            return
        if not result.destroyed:
            return
        if m.instance_id:
            self._retired_instances.add(m.instance_id)
        for run in on:
            if self._runs.get(run.job_id) is not run:
                continue
            pulled = result.pulled.get(run.remote_id)
            run_outcome, detail = details.get(run.job_id, (outcome, reason))
            try:
                if run.phase == "delivering" and pulled is not None and (run.exit_code == 0 or run.checked):
                    # It finished; deliver from what retire saved.
                    run.fetched = pulled
                    run.checked = True
                    run.fetch_failures = 0
                    continue
                if run.remote_id in result.lost:
                    detail += "; its outputs could not be saved"
                    self.obs.emit(
                        ev.QUEUE_OUTPUTS_LOST, job_id=run.job_id, attempt=run.number,
                        slot=m.slot, remote_id=run.remote_id, reason=reason,
                    )
                self._fault(run, run_outcome, detail, outputs_dir=pulled)
            except _Dropped:
                pass

    def _watch(self) -> None:
        tracked = {m.slot: m for m in self.pool.machines()}
        listings: dict[str, Optional[dict[str, dict]]] = {}
        for run in self._snapshot():
            if self._runs.get(run.job_id) is not run or run.phase == "pending":
                continue
            m = tracked.get(run.slot or "")
            try:
                if run.phase == "delivering":
                    answering = m is not None and m.state in _ANSWERING
                    if answering:
                        self._keep_deadline(run, m)
                    if run.fetched is not None or answering:
                        self._deliver(run, m if answering else None)
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
                self._follow(run, m, listed.get(run.remote_id))
            except _Dropped:
                continue
            except _Demoted:
                raise
            except Exception as exc:  # noqa: BLE001 - one job never stops the tick
                self.obs.emit(ev.PROVIDER_ERROR, operation="watch", job_id=run.job_id, error=str(exc)[:500])
            self._heartbeat()

    def _keep_deadline(self, run: _Run, m: PoolMachine) -> None:
        """A machine holding outputs that are not delivered yet keeps its deadline ahead."""
        margin = run.limits.deadline_margin_s if run.limits and run.limits.deadline_margin_s else self.default_margin_s
        now = self.clock()
        if m.deadline_at is not None and m.deadline_at - now >= margin / 2:
            return
        try:
            self.pool.extend_deadline(m, max(now + margin, m.deadline_at or 0.0))
        except Exception as exc:  # noqa: BLE001
            self.obs.emit(ev.PROVIDER_ERROR, operation="extend_deadline", slot=m.slot, error=str(exc)[:500])
            return
        lim = dict(run.job.limits or {})
        lim["deadline_at"] = max(float(lim.get("deadline_at") or 0.0), float(m.deadline_at or now + margin))
        self._update(run, limits=lim)

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
        if run.phase == "cancelling" or (state == "cancelled" and run.job.cancel_requested):
            self._cancelled(run, m)
            return
        if state == "cancelled":
            # Nobody here asked for it (a coordinator that lost its lease cancels its run).
            self._fault(run, Outcome.MACHINE_LOST, f"the run was cancelled on machine {m.slot}")
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
        the job DELIVERING and is tried again next tick, up to MAX_DELIVERY_FAILURES."""
        if run.fetched is None:
            if m is None:
                return
            dest = self._dir(run) / f"out-a{run.number}"
            try:
                shutil.rmtree(dest, ignore_errors=True)
                self.pool.fetch_outputs(m, run.remote_id, dest)
            except Exception as exc:  # noqa: BLE001
                self._fetch_failed(run, m, "fetch", [f"{type(exc).__name__}: {str(exc)[:300]}"])
                return
            run.fetched = dest
            run.fetch_failures, run.last_fetch_problem = 0, None
            self._write_marker(run)
            self._heartbeat()

        ok_exit = run.exit_code == 0 and run.remote_state in (None, "succeeded")
        if not run.checked:
            health = None
            if m is not None and m.state in _ANSWERING and (not ok_exit or self.queue.health_check.between_jobs):
                try:
                    health = self.pool.health_check(m)
                except Exception as exc:  # noqa: BLE001 - machine-side: counts like a failed fetch
                    self._fetch_failed(run, m, "health_check", [f"{type(exc).__name__}: {str(exc)[:300]}"])
                    return
                self._heartbeat()
            run.checked = True  # only once the check has returned
            self._write_marker(run)
            if health is not None and not health.ok:
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
                m = None
            elif health is not None and run.remote_state == "lost":
                self._fault(
                    run, Outcome.MACHINE_LOST, "the job's process vanished without an exit code",
                    outputs_dir=run.fetched,
                )
                return

        now = self.clock()
        if run.next_store_at is not None and now < run.next_store_at:
            return  # backing off from a store that failed
        try:
            ref = self.outputs.put_outputs(run.job_id, run.number, run.fetched, success=ok_exit)
            problems = self.outputs.verify(ref)
        except Exception as exc:  # noqa: BLE001
            ref, problems = None, [f"{type(exc).__name__}: {str(exc)[:300]}"]
        if problems or ref is None:
            self._store_failed(run, problems)
            return
        run.store_failures, run.next_store_at, run.last_store_problem = 0, None, None
        self._stored(run, ref, run.fetched)

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

    def _store_failed(self, run: _Run, problems: list[str]) -> None:
        """The store is the problem, not the machine: keep the local copy, back off and
        try again; never retire anything for it."""
        run.store_failures += 1
        run.next_store_at = self.clock() + min(self.tick_s * 2 ** (run.store_failures - 1), STORE_BACKOFF_MAX_S)
        summary = "; ".join(problems)[:500]
        if summary != run.last_store_problem:
            run.last_store_problem = summary
            self.obs.emit(
                ev.QUEUE_OUTPUTS_DELIVERY_FAILED, job_id=run.job_id, attempt=run.number,
                stage="store", problems=list(problems)[:10], failures=run.store_failures,
                retry_in_s=round(run.next_store_at - self.clock(), 1),
            )

    def _fetch_failed(self, run: _Run, m: Optional[PoolMachine], stage: str, problems: list[str]) -> None:
        run.fetch_failures += 1
        summary = f"{stage}: " + "; ".join(problems)[:500]
        if summary != run.last_fetch_problem:
            run.last_fetch_problem = summary
            self.obs.emit(
                ev.QUEUE_OUTPUTS_DELIVERY_FAILED, job_id=run.job_id, attempt=run.number,
                stage=stage, problems=list(problems)[:10], failures=run.fetch_failures,
            )
        if run.fetch_failures < MAX_FETCH_FAILURES or m is None or m.state not in _ANSWERING:
            return
        # The machine leaves; it pulls what it can. Pulled -> delivered from there,
        # otherwise the attempt ends with the loss recorded.
        run.checked = True  # leaving for delivery reasons says nothing against the job's exit
        detail = f"outputs could not be delivered after {run.fetch_failures} tries ({stage}: {'; '.join(problems)[:200]})"
        self._retire(
            m, Outcome.UNHEALTHY, "outputs could not be delivered", True,
            details={run.job_id: (Outcome.DELIVERY_FAILED, detail)},
        )

    def _stored(self, run: _Run, ref: OutputRef, src: Optional[Path] = None) -> None:
        self._stored_event(run.job_id, run.number, ref)
        if src is not None:
            self._mark_stored(run.job_id, Path(src))

    def _stored_event(self, job_id: str, attempt: int, ref: OutputRef) -> None:
        total = sum(int(f.get("bytes") or 0) for f in ref.files)
        digest = hashlib.sha256(json.dumps(ref.files, sort_keys=True).encode()).hexdigest()
        self.obs.inc("gpu_queue_outputs_bytes_total", total)
        self.obs.emit(
            ev.QUEUE_OUTPUTS_STORED, job_id=job_id, attempt=attempt, files=len(ref.files),
            bytes=total, sha256=digest, location=ref.location,
        )

    def _mark_stored(self, job_id: str, src: Path) -> None:
        """``out-a<n>.stored`` beside an out-a<n> of ours: its files are in the store."""
        if src.parent != self.workdir / job_id or not _OUT.fullmatch(src.name):
            return  # not one of our local copies (e.g. a directory retire() pulled into)
        try:
            src.with_name(src.name + ".stored").touch()
        except OSError:
            pass

    def _store_attempt(self, run: _Run, src: Path) -> bool:
        """Keep a failed attempt's outputs (for the next attempt's $JAMBU_PREVIOUS_OUTPUT_DIR).
        True when there was nothing to keep or it is stored and verified."""
        if not src.is_dir() or not any(p.is_file() for p in src.rglob("*")):
            return True
        try:
            ref = self.outputs.put_outputs(run.job_id, run.number, src, success=False)
            problems = self.outputs.verify(ref)
        except Exception as exc:  # noqa: BLE001
            ref, problems = None, [f"{type(exc).__name__}: {str(exc)[:300]}"]
        if problems or ref is None:
            self.obs.emit(
                ev.QUEUE_OUTPUTS_DELIVERY_FAILED, job_id=run.job_id, attempt=run.number,
                stage="store_attempt", problems=list(problems)[:10],
            )
            return False
        self._stored(run, ref, src)
        self._update(run, outputs_ref=asdict(ref))
        return True

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
        # An attempt sent back by `drain --now` is nobody's fault: it does not count toward the cap.
        if sum(1 for a in attempts if a.outcome != Outcome.DRAINED) >= self.queue.max_attempts:
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
        self._cleanup(run)

    def _reject(self, run: _Run, reason: str) -> None:
        now = self.clock()
        self.obs.emit(ev.QUEUE_PREFLIGHT_REJECTED, job_id=run.job_id, attempt=run.number, reason=reason)
        attempt = Attempt(
            number=run.number, machine=None, remote_id=run.remote_id, started_at=now, ended_at=now,
            outcome=Outcome.PREFLIGHT_REJECTED, detail=reason[:500],
        )
        self._settle(run, JobState.FAILED, {"outcome": Outcome.PREFLIGHT_REJECTED.value, "reason": reason}, attempt)

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
        self._store_leftovers(only=run.job_id)  # earlier attempts' outputs, if the store is back
        self._cleanup(run)

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
            1 for r in self._snapshot()
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
        try:
            self._place_inner()
        except _SkipPlacement:
            pass

    def _place_inner(self) -> None:
        with self._lock:
            paused = self.backend.paused()
        if self._adopt:
            self._check_lead()
            self._retry_adoptions()
        for run in sorted(
            (r for r in self._snapshot() if r.phase == "pending"),
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
                elif run.job.cancel_requested and run.slot is None:
                    self._settle(run, JobState.CANCELLED, {"outcome": "cancelled", "reason": "cancelled before it started"})
            except _Dropped:
                continue
            self._heartbeat()
        if paused:
            return
        for m in self.pool.machines():
            while self._free(m) > 0:
                self._check_lead()
                with self._lock:
                    if not self.leader:
                        raise _Demoted()
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
            if run.slot is not None:
                # A dispatch was tried: it may have started.
                try:
                    self.pool.cancel(m, run.remote_id)
                except Exception:  # noqa: BLE001 - best effort
                    pass
            self._settle(run, JobState.CANCELLED, {"outcome": "cancelled", "reason": "cancelled before it started"})
            return
        if run.limits is None:
            try:
                ctx = PreflightContext(
                    hourly_usd=float(m.hourly_usd or self.pool.next_offer_usd() or 0.0),
                    card_vram_gb=self.pool.card_vram_gb(),
                    defaults=self.queue.defaults,
                    attempt=run.number,
                )
                verdict = self.preflight(job, ctx)
            except Exception as exc:  # noqa: BLE001 - the rest of the tick still runs
                self._error_once("preflight", exc, job_id=job.id)
                raise _SkipPlacement() from None
            self._ok("preflight")
            if not verdict.ok:
                self._reject(run, verdict.reason)
                return
            run.limits = verdict.limits
        limits = run.limits
        until = now + limits.timeout_s + limits.deadline_margin_s
        try:
            self.pool.extend_deadline(m, max(until, m.deadline_at or 0.0))
        except Exception as exc:  # noqa: BLE001 - no deadline, no dispatch; next tick
            self.obs.emit(ev.PROVIDER_ERROR, operation="extend_deadline", slot=m.slot, error=str(exc)[:500])
            return
        until = max(until, float(m.deadline_at or 0.0))

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
            "instance_id": m.instance_id, "owner": self.worker_id, "worker": self.worker_id,
            "started_at": run.started_at, "deadline_at": until,
        }
        self._check_lead()
        self._update(run, limits=lim, machine=m.slot)
        run.slot = m.slot  # from here a retry goes to this machine only
        self._check_lead()
        try:
            outcome = self.pool.dispatch(m, run.remote_id, bundle, manifest, env)
        except Exception as exc:  # noqa: BLE001 - "already_running" next time if it did start
            run.dispatch_failures += 1
            self.obs.emit(
                ev.PROVIDER_ERROR, operation="dispatch", job_id=job.id, slot=m.slot,
                error=str(exc)[:500], failures=run.dispatch_failures,
            )
            if run.dispatch_failures >= MAX_DISPATCH_FAILURES:
                detail = f"dispatch failed {run.dispatch_failures} times on {m.slot}: {str(exc)[:200]}"
                self._retire(m, Outcome.UNHEALTHY, "dispatch kept failing", True,
                             details={run.job_id: (Outcome.UNHEALTHY, detail)})
            return
        try:
            if self._runs.get(run.job_id) is not run:
                raise _Dropped()  # the lease went while the dispatch was on its way
            run.phase = "running"
            self._update(run, state=JobState.RUNNING)
        except _Dropped:
            if outcome == "started":
                self._cancel_orphan(m, run)
            raise
        self._report.dispatched += 1
        wait_s = max(0.0, now - float(job.submitted_at))
        self.obs.timing("gpu_queue_job_wait_seconds", wait_s)
        self.obs.emit(
            ev.QUEUE_JOB_STARTED, job_id=job.id, attempt=run.number, slot=m.slot, remote_id=run.remote_id,
            dispatch=outcome, wait_s=round(wait_s, 3), timeout_s=limits.timeout_s, deadline_at=until,
            previous_outputs=bool(env),
        )

    def _cancel_orphan(self, m: PoolMachine, run: _Run) -> None:
        """We started a run but no longer hold its job: whoever does may start it elsewhere."""
        try:
            self.pool.cancel(m, run.remote_id)
            done = True
        except Exception:  # noqa: BLE001 - best effort; the auditor (I2) catches what is left
            done = False
        self.obs.emit(
            ev.QUEUE_STALE_WRITE_REFUSED, job_id=run.job_id, operation="dispatch",
            worker=self.worker_id, remote_id=run.remote_id, slot=m.slot, cancelled=done,
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

    def _queued_in_flight(self) -> int:
        """Queued jobs that are runs waiting to be re-adopted, not new work."""
        n = 0
        with self._lock:
            ids = list(self._adopt)
        for job_id in ids:
            with self._lock:
                job = self.backend.get(job_id)
            if job is not None and job.state == JobState.QUEUED:
                n += 1
        return n

    def _scale(self) -> None:
        q = self.queue
        now = self.clock()
        with self._lock:
            paused = self.backend.paused()
            self._stats = stats = self.backend.stats()
        machines = self.pool.machines()
        queued = 0
        if not paused:
            queued = max(0, int(stats.by_state.get(JobState.QUEUED.value, 0)) - self._queued_in_flight())
            queued += sum(1 for r in self._snapshot() if r.phase == "pending" and r.slot is None and not r.locate)
        views = [MachineView(m.slot, m.state, m.hourly_usd, self._occupied(m.slot)) for m in machines]
        counted = sum(1 for v in views if v.state in ("provisioning", "checking", "ready", "busy"))
        min_machines = 0 if paused else q.min_machines  # a paused (or drained) queue keeps no warm machine
        offer = None
        if queued > 0 or counted < min_machines:
            try:
                offer = self.pool.next_offer_usd()
                self._ok("next_offer")
            except Exception as exc:  # noqa: BLE001
                self._error_once("next_offer", exc)
        oldest = stats.oldest_queued_age_s if queued else None
        if oldest is None and queued:
            oldest = 0.0
        inp = ScaleInput(
            now=now, queued=queued, oldest_queued_age_s=oldest, machines=views,
            pool_usd_per_hour=pool_usd_per_hour(machines), next_offer_usd=offer,
            min_machines=min_machines, max_machines=q.max_machines, max_usd_per_hour=q.max_usd_per_hour,
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
            if not self.leader:
                break
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
                with self._lock:
                    self.obs.gauge("gpu_queue_paused", 1.0 if self.backend.paused() else 0.0)
                    stats = self._stats or self.backend.stats()
                self.obs.gauge("gpu_queue_depth", float(stats.by_state.get(JobState.QUEUED.value, 0)))
                self.obs.gauge("gpu_queue_oldest_queued_seconds", float(stats.oldest_queued_age_s or 0.0))
                pending = [r for r in self._snapshot() if r.phase == "delivering"]
                self.obs.gauge("gpu_queue_outputs_pending", float(len(pending)))
                oldest = max((now - (r.delivering_since or now) for r in pending), default=0.0)
                self.obs.gauge("gpu_queue_outputs_oldest_pending_seconds", oldest)
            self._stats = None
            self.obs.write()
        except Exception as exc:  # noqa: BLE001 - metrics never stop the loop
            self.obs.emit(ev.PROVIDER_ERROR, operation="metrics", error=str(exc)[:500])
