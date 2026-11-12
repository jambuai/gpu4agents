"""The auditor: checks the guarantees of queue mode and repairs what it safely can.

Codes (each found violation emits ``queue_invariant_violated`` and counts in
``gpu_queue_invariant_violations_total{code}``):

- ``I1_running_without_machine``: a RUNNING job whose run is surely on no machine, by
  the coordinator's own adoption rules (a machine this pool cannot see is waited for
  until its deadline plus a grace). Repair: requeue MACHINE_LOST.
- ``I2_duplicate_execution``: a remote run going on that the queue does not place
  there (the job is final, gone, or in flight elsewhere or as another attempt).
  Repair: cancel the remote run.
- ``I3_outputs_unverified``: a success whose recorded outputs fail verification.
  Repair: a settle that never landed goes back to DELIVERING; a settled job is reported.
- ``I4_orphan_machine``: a provider instance labeled as this pool's (prefix and
  fingerprint) that no slot tracks. Repair: retire it, unless it is the machine of a
  job not final whose deadline plus the grace has not passed.
- ``I5_spend_over_cap``: the pool costs more per hour than ``max_usd_per_hour``. Alert.
- ``I6_job_past_deadline``: a RUNNING job past its deadline. Repair: cancel it on the
  machine and requeue TIMEOUT.
- ``I7_stale_coordinator``: the lease holder's last heartbeat is older than three
  ticks, or no coordinator holds the lease while jobs are not final. Alert.
- ``I8_outputs_not_stored``: outputs fetched to ``<state_dir>/queue/work/<job>/out-a<n>``
  more than an hour ago and not marked stored. Alert.

Who may repair what. Changing a job needs its claim token, which only the coordinator
holds, so job repairs (I1, I3, I6) go through the coordinator passed in; without one
they are reported with ``repaired=False``. Machine repairs (I2, I4) run inside the
coordinator, or standalone only while no coordinator holds the lease.

The same (code, job, slot) is reported once per change (it appears, or its repaired
flag changes) across the audits that share an ``AuditMemory``; the metric counts every
detection.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core import events as ev
from ..core.config import QueueConfig
from ..core.durations import parse_duration
from .coordinator import (
    _ANSWERING,
    COORDINATOR_LEASE_TICKS,
    DEADLINE_GRACE_S,
    MIN_COORDINATOR_LEASE_S,
    Coordinator,
    find_run,
    lost_detail,
)
from .models import FINAL_STATES, JobState, QueueJob
from .observe import Observer
from .outputs import OutputRef
from .spend import pool_usd_per_hour

METRIC = "gpu_queue_invariant_violations_total"
STALE_AFTER_TICKS = 3
OUTPUTS_UNSTORED_ALERT_S = 3600.0
MAX_VERIFY_PER_AUDIT = 100  # settled successes verified per audit; each is verified once
_ACTIVE = {JobState.QUEUED, JobState.CLAIMED, JobState.RUNNING, JobState.DELIVERING}
_OUT = re.compile(r"out-a([0-9]+)")
_REMOTE = re.compile(r"(.+)-a([0-9]+)")


@dataclass
class Violation:
    code: str
    job_id: Optional[str]
    slot: Optional[str]
    detail: str
    repaired: bool


@dataclass
class AuditMemory:
    """What one auditor remembers between audits."""

    reported: dict[tuple, bool] = field(default_factory=dict)  # key -> repaired, last audit
    verified: dict[str, tuple[str, list[str]]] = field(default_factory=dict)  # job -> (location, problems)


def audit(
    backend: Any,
    pool: Any,
    outputs: Any,
    observer: Observer,
    *,
    repair: bool,
    coordinator: Optional[Coordinator] = None,
    config: Any = None,
    clock: Optional[Callable[[], float]] = None,
    memory: Optional[AuditMemory] = None,
) -> list[Violation]:
    """Check every guarantee once; repair what is safe when ``repair``.

    ``config`` (a RuntimeConfig) and ``clock`` are taken from the coordinator when one
    is given; standalone they default to ``pool.config`` and ``backend.now``."""
    return _Audit(backend, pool, outputs, observer, repair, coordinator, config, clock, memory).run()


def _num(lim: dict, key: str) -> Optional[float]:
    try:
        value = lim.get(key)
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _ended_by(lim: dict, default_margin_s: float) -> Optional[float]:
    """When a run recorded in ``lim`` has surely ended on its machine: its deadline plus
    the grace; else started + timeout + margin plus the grace; else unknown."""
    deadline = _num(lim, "deadline_at")
    if deadline is not None:
        return deadline + DEADLINE_GRACE_S
    started, timeout = _num(lim, "started_at"), _num(lim, "timeout_s")
    if started is None or timeout is None:
        return None
    margin = _num(lim, "deadline_margin_s")
    return started + timeout + (default_margin_s if margin is None else margin) + DEADLINE_GRACE_S


class _Audit:
    def __init__(self, backend, pool, outputs, observer, repair, coordinator, config, clock, memory) -> None:
        self.backend = backend
        self.pool = pool
        self.outputs = outputs
        self.obs = observer
        self.coord = coordinator
        if coordinator is not None:
            cfg = coordinator.config
            self.queue: QueueConfig = coordinator.queue
            self.clock = coordinator.clock
            self.lock: Any = coordinator._lock
            self.workdir: Optional[Path] = coordinator.workdir
            self.memory = memory or coordinator.audit_memory
        else:
            cfg = config if config is not None else getattr(pool, "config", None)
            self.queue = getattr(cfg, "queue", None) or QueueConfig()
            self.clock = clock or backend.now
            self.lock = contextlib.nullcontext()
            self.workdir = Path(cfg.state_dir) / "queue" / "work" if cfg is not None else None
            self.memory = memory or AuditMemory()
        self.repair = repair
        self.tick_s = self.queue.tick_s
        self.default_margin_s = parse_duration(self.queue.defaults.deadline_margin) or 600.0
        self.found: list[tuple[Violation, tuple]] = []
        self._listings: dict[str, Optional[list[dict]]] = {}

    # ------------------------------------------------------------------

    def run(self) -> list[Violation]:
        now = self.clock()
        with self.lock:
            lease = self.backend.coordinator()
            active = self.backend.list(_ACTIVE, limit=100_000)
        leader = self.coord is not None and self.coord.leader
        self.fix_jobs = self.repair and leader
        self.fix_machines = self.repair and (leader if self.coord is not None else lease is None)

        self._i7(lease, active)
        self._i2()
        self._running(active, now)
        self._i3(active)
        self._i4(active, now)
        self._i5()
        self._i8(now)
        return self._report()

    def _add(self, code: str, job_id: Optional[str], slot: Optional[str], detail: str, repaired: bool,
             subject: Optional[str] = None) -> None:
        v = Violation(code=code, job_id=job_id, slot=slot, detail=detail[:500], repaired=bool(repaired))
        self.found.append((v, (code, job_id, slot, subject)))

    def _report(self) -> list[Violation]:
        self.found.sort(key=lambda f: (f[0].code, f[0].job_id or "", f[0].slot or "", f[1][3] or ""))
        current: dict[tuple, bool] = {}
        for v, key in self.found:
            self.obs.inc(METRIC, code=v.code)
            if self.memory.reported.get(key) != v.repaired and current.get(key) != v.repaired:
                self.obs.emit(
                    ev.QUEUE_INVARIANT_VIOLATED, code=v.code, job_id=v.job_id, slot=v.slot,
                    detail=v.detail, repaired=v.repaired,
                )
            current[key] = v.repaired
        self.memory.reported = current
        return [v for v, _ in self.found]

    def _list_jobs(self, m) -> list[dict]:
        """The machine's remote runs, listed once per audit (until a repair changes them)."""
        if m.slot not in self._listings:
            try:
                self._listings[m.slot] = list(self.pool.remote_jobs(m))
            except Exception:  # noqa: BLE001 - cannot tell: the pool marks it in time
                self._listings[m.slot] = None
        listed = self._listings[m.slot]
        if listed is None:
            raise RuntimeError(f"machine {m.slot} does not list its jobs")
        return listed

    def _changed(self) -> None:
        self._listings.clear()

    def _get(self, job_id: str) -> Optional[QueueJob]:
        with self.lock:
            return self.backend.get(job_id)

    # ------------------------------------------------------------------ I7

    def _i7(self, lease: Optional[dict], active: list[QueueJob]) -> None:
        stale_s = STALE_AFTER_TICKS * self.tick_s
        if lease is None:
            if active:
                self._add(
                    "I7_stale_coordinator", None, None,
                    f"no coordinator holds the lease while {len(active)} job(s) are not final", False,
                )
            return
        lease_s = max(COORDINATOR_LEASE_TICKS * self.tick_s, MIN_COORDINATOR_LEASE_S)
        try:
            age = self.backend.now() - (float(lease["lease_until"]) - lease_s)
        except (KeyError, TypeError, ValueError):
            return
        if age > stale_s:
            self._add(
                "I7_stale_coordinator", None, None,
                f"coordinator {lease.get('owner')}: last heartbeat {age:.0f}s ago (more than {stale_s:g}s)", False,
            )

    # ------------------------------------------------------------------ I2

    def _legit(self, job: Optional[QueueJob], remote_id: str, m) -> bool:
        """The queue places this run here: the job is not final and its in-flight record
        names this remote id on this machine."""
        if job is None or job.state in FINAL_STATES or not Coordinator._in_flight(job):
            return False
        lim = job.limits or {}
        if lim.get("remote_id") != remote_id:
            return False
        instance = lim.get("instance_id")
        return (m.instance_id == instance) if instance else (lim.get("slot") == m.slot)

    def _i2(self) -> None:
        for m in self.pool.machines():
            if m.state not in _ANSWERING:
                continue
            try:
                listed = self._list_jobs(m)
            except RuntimeError:
                continue
            for rec in listed:
                rid = str(rec.get("id") or "")
                match = _REMOTE.fullmatch(rid)
                if rec.get("state") != "running" or match is None:
                    continue  # not running, or not a queue run
                job = self._get(match.group(1))
                if self._legit(job, rid, m):
                    continue
                if job is None:
                    why = "the queue has no such job"
                elif job.state in FINAL_STATES:
                    why = f"the job is {job.state.value}"
                else:
                    lim = job.limits or {}
                    where = lim.get("remote_id") if Coordinator._in_flight(job) else None
                    why = f"the job is {job.state.value}" + (
                        f" with {where} in flight on {lim.get('slot')}" if where else " with nothing in flight"
                    )
                repaired = False
                if self.fix_machines:
                    try:
                        self.pool.cancel(m, rid)
                        repaired = True
                        self._changed()
                    except Exception as exc:  # noqa: BLE001 - the next audit tries again
                        why += f"; cancel failed: {type(exc).__name__}: {str(exc)[:200]}"
                self._add(
                    "I2_duplicate_execution", match.group(1), m.slot,
                    f"{rid} runs on {m.slot} but {why}", repaired, subject=rid,
                )

    # ------------------------------------------------------------------ I1, I6

    def _lost(self, job: QueueJob, now: float) -> Optional[str]:
        if self.coord is not None:
            return self.coord.lost(job)
        lim = job.limits or {}
        home, m, _ = find_run(self.pool.machines(), lim, str(lim.get("remote_id") or ""), self._list_jobs)
        if m is not None:
            return None
        return lost_detail(home, _num(lim, "started_at"), now, lambda: _ended_by(lim, self.default_margin_s))

    def _running(self, active: list[QueueJob], now: float) -> None:
        for job in active:
            if job.state != JobState.RUNNING or not Coordinator._in_flight(job):
                continue
            lim = job.limits or {}
            slot = lim.get("slot") or job.machine
            detail = self._lost(job, now)
            if detail is not None:
                repaired = self.fix_jobs and self.coord.requeue_lost(job.id)
                if repaired:
                    self._changed()
                self._add("I1_running_without_machine", job.id, slot, f"{lim.get('remote_id')}: {detail}", repaired)
                continue
            deadline = _num(lim, "deadline_at")
            if deadline is not None and now > deadline:
                repaired = self.fix_jobs and self.coord.cancel_overdue(job.id)
                if repaired:
                    self._changed()
                self._add(
                    "I6_job_past_deadline", job.id, slot,
                    f"{lim.get('remote_id')} still running {now - deadline:.0f}s past its deadline", repaired,
                )

    # ------------------------------------------------------------------ I3

    def _problems(self, ref: Optional[dict]) -> Optional[list[str]]:
        """Verification problems, [] when intact, None when the store cannot tell now."""
        if not ref:
            return ["no outputs recorded"]
        try:
            return list(self.outputs.verify(OutputRef(**ref)))
        except Exception:  # noqa: BLE001 - the store is down: the next audit asks again
            return None

    def _i3(self, active: list[QueueJob]) -> None:
        for job in active:
            settling = (job.limits or {}).get("settling")
            if not Coordinator._unsettled(job) or settling.get("state") != JobState.SUCCEEDED.value:
                continue
            problems = self._problems(job.outputs_ref)
            if not problems:
                continue
            repaired = self.fix_jobs and self.coord.redeliver(job.id)
            if repaired:
                self._changed()
            self._add(
                "I3_outputs_unverified", job.id, job.machine,
                "a success not settled yet whose outputs fail verification: " + "; ".join(problems), repaired,
            )
        with self.lock:
            done = self.backend.list({JobState.SUCCEEDED}, limit=100_000)
        budget = MAX_VERIFY_PER_AUDIT
        for job in done:
            location = str((job.outputs_ref or {}).get("location") or "")
            seen = self.memory.verified.get(job.id)
            if seen is None or seen[0] != location:
                if budget <= 0:
                    continue
                budget -= 1
                problems = self._problems(job.outputs_ref)
                if problems is None:
                    continue
                seen = self.memory.verified[job.id] = (location, problems)
            if seen[1]:
                self._add(
                    "I3_outputs_unverified", job.id, job.machine,
                    "settled as succeeded but its outputs fail verification (a final job is not reopened): "
                    + "; ".join(seen[1]),
                    False,
                )

    # ------------------------------------------------------------------ I4

    def _i4(self, active: list[QueueJob], now: float) -> None:
        orphans = getattr(self.pool, "orphans", None)
        if orphans is None:
            return
        try:
            found = list(orphans())
        except Exception as exc:  # noqa: BLE001
            self.obs.emit(ev.PROVIDER_ERROR, operation="audit_orphans", error=str(exc)[:500])
            return
        # A job's in-flight machine is never retired before its deadline plus the grace.
        protected: dict[str, str] = {}
        for job in active:
            lim = job.limits or {}
            instance = lim.get("instance_id")
            if not instance:
                continue
            until = _ended_by(lim, self.default_margin_s)
            if until is None or now < until:
                protected[str(instance)] = job.id
        for inst in found:
            label = getattr(inst, "label", "") or ""
            owner = protected.get(inst.id)
            if owner is not None:
                self._add(
                    "I4_orphan_machine", owner, None,
                    f"instance {inst.id} ({label}) is untracked but is job {owner}'s machine until its deadline "
                    "plus the grace; not retired",
                    False, subject=inst.id,
                )
                continue
            detail = f"instance {inst.id} ({label}) is labeled as this pool's but no slot tracks it"
            repaired = False
            if self.fix_machines:
                try:
                    self.pool.retire_instance(inst, "orphan: no slot tracks it (auditor)")
                    repaired = True
                except Exception as exc:  # noqa: BLE001 - the next audit tries again
                    detail += f"; retire failed: {type(exc).__name__}: {str(exc)[:200]}"
            self._add("I4_orphan_machine", None, None, detail, repaired, subject=inst.id)

    # ------------------------------------------------------------------ I5

    def _i5(self) -> None:
        cap = self.queue.max_usd_per_hour
        if cap is None:
            return
        spend = pool_usd_per_hour(self.pool.machines())
        if spend > cap + 1e-9:
            self._add(
                "I5_spend_over_cap", None, None,
                f"the pool costs ${spend:.4f}/h, over max_usd_per_hour ${cap:g}/h", False,
            )

    # ------------------------------------------------------------------ I8

    def _i8(self, now: float) -> None:
        if self.workdir is None or not self.workdir.is_dir():
            return
        for base in sorted(p for p in self.workdir.iterdir() if p.is_dir()):
            old = []
            for child in base.iterdir():
                match = _OUT.fullmatch(child.name)
                if match is None or not child.is_dir() or (base / f"{child.name}.stored").exists():
                    continue
                try:
                    age = now - child.stat().st_mtime
                except OSError:
                    continue
                if age > OUTPUTS_UNSTORED_ALERT_S:
                    old.append((int(match.group(1)), age))
            if old:
                parts = ", ".join(f"attempt {n} ({age / 3600:.1f} h)" for n, age in sorted(old))
                self._add(
                    "I8_outputs_not_stored", base.name, None,
                    f"fetched outputs not in the output store after more than an hour: {parts}", False,
                )
