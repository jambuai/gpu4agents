"""The pool: the machines queue mode rents from one gpu.yaml.

Each machine is an Engine in its own slot (``s1``, ``s2``, ...) with its own
state under ``<state_dir>/pool/<slot>/`` and the label
``jambu-<name>-<slot>-<fp6>``. The pool opens machines without blocking (the
rent, the boot and the first health check run in a thread), watches them on
every ``poll()``, pushes deadlines to their watchdogs and retires the ones
that fail.

States: provisioning -> checking -> ready <-> busy, and then unhealthy,
unresponsive, gone or retired. ``unresponsive`` means the provider says the
machine runs but its watchdog has not answered for ``unresponsive_after``.
Nothing on the machine changes for the pool: health checks are ordinary jobs
on the job runner, deadlines go through the watchdog's ``/deadline`` route.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional

from ..core import events as ev
from ..core.beacon import WatchdogClient, WatchdogUnavailable
from ..core.config import QueueConfig, RuntimeConfig
from ..core.durations import parse_duration
from ..core.engine import Engine
from ..core.errors import (
    BudgetExceededError,
    CapacityUnavailableError,
    JambuError,
    ProviderError,
    StateError,
)
from ..core.events import EventLog
from ..core.hosts import BadHosts
from ..core.job import JobClient, JobConflict
from ..core.models import JAMBU_LABEL_PREFIX, Instance, InstanceState
from ..core.state import RuntimeState
from ..providers.base import GPUProvider
from .observe import Observer
from .pool_types import MachineState, PoolMachine
from .spend import SpendLedger, machine_usd_per_hour, pool_usd_per_hour

HEALTH_JOB_PREFIX = "pool-health-"
_SLOT = re.compile(r"s[0-9]+")
BACKOFF_FIRST_S = 30.0
BACKOFF_MAX_S = 300.0
OFFER_CACHE_S = 60.0
# How long a health job may take beyond its own timeout before the pool gives up on it.
HEALTH_SLACK_S = 30.0
# Bring-up attempts for one slot in this process before the machine is given up as unhealthy.
MAX_RESUMES = 3
# A labeled instance younger than this may still be on its way into a slot's records.
ORPHAN_MIN_AGE_S = 120.0

_COUNTED = (MachineState.PROVISIONING, MachineState.CHECKING, MachineState.READY, MachineState.BUSY)
# Everything that may still be billing at full rate counts toward max_machines.
_HELD = _COUNTED + (MachineState.DRAINING, MachineState.UNHEALTHY, MachineState.UNRESPONSIVE, MachineState.HELD)
_BOOTING = (MachineState.PROVISIONING, MachineState.CHECKING)
_WATCHED = (MachineState.READY, MachineState.BUSY, MachineState.DRAINING, MachineState.HELD)

_TORCH_CHECK = (
    'python3 -c "import torch; assert torch.cuda.is_available(); torch.ones(1).cuda().sum().item()"'
)


def builtin_health_command(command: Optional[str], min_free_vram_gb: Optional[float] = None) -> str:
    """The health check's shell command: the project's own, else a small CUDA operation
    (``nvidia-smi -L`` where torch is absent), plus the free-memory floor when set."""
    if command:
        base = command
    else:
        base = (
            'if python3 -c "import torch" >/dev/null 2>&1; then '
            f"{_TORCH_CHECK}; else nvidia-smi -L; fi"
        )
    if not min_free_vram_gb:
        return base
    need_mib = int(round(float(min_free_vram_gb) * 1024))
    vram = (
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | "
        f"awk -v need={need_mib} "
        "'{ if ($1 + 0 < need) { print \"GPU \" NR-1 \" has \" $1 \" MiB free, needs \" need; bad=1 } } "
        "END { exit bad }'"
    )
    return f"( {base} ) && {vram}"


@dataclass
class HealthResult:
    ok: bool
    detail: str


@dataclass
class RetireResult:
    """What retire() saved from the machine before destroying it."""

    pulled: dict[str, Path] = field(default_factory=dict)  # remote id -> local directory
    lost: list[str] = field(default_factory=list)  # remote ids whose outputs could not be saved
    destroyed: bool = False


class _SlotEngine(Engine):
    """An Engine that tells the pool the moment its watchdog answers, so the machine
    gets a deadline before anything else happens on it."""

    on_watchdog: Optional[Callable[["_SlotEngine", RuntimeState], None]] = None

    def _connect_watchdog(self, state: RuntimeState) -> None:
        super()._connect_watchdog(state)
        if self.on_watchdog is not None and state.watchdog.installed and not state.watchdog.last_error:
            self.on_watchdog(self, state)


class Pool:
    def __init__(
        self,
        config: RuntimeConfig,
        observer: Observer,
        provider: Optional[GPUProvider] = None,
        clock: Callable[[], float] = time.time,
        *,
        state_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.queue: QueueConfig = config.queue or QueueConfig()
        self.obs = observer
        self.clock = clock
        self.state_dir = Path(state_dir) if state_dir is not None else config.state_dir
        self.dir = self.state_dir / "pool"
        # Builds specs and labels, finds offers; never rents or saves.
        self._probe = Engine(config, provider=provider, state_dir=self.dir / ".probe")
        self.provider = self._probe.provider
        self.spend = SpendLedger(self.dir / "spend.jsonl")

        self._lock = threading.RLock()
        self._machines: dict[str, PoolMachine] = {}
        self._engines: dict[str, _SlotEngine] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._billed_at: dict[str, float] = {}
        self._deadline_set: set[str] = set()  # slots whose watchdog got a deadline from this pool

        self.last_open_at: Optional[float] = None
        # Backoff runs from the moment an open failed, not from when it began.
        self.last_failed_at: Optional[float] = None
        self.open_backoff_s: float = 0.0
        self._resumes: dict[str, int] = {}
        self._seen: Optional[dict[str, float]] = None  # untracked instance id -> first seen
        self._blocked: Optional[str] = None
        self._offer: Optional[Any] = None
        self._offer_at: Optional[float] = None
        self._load()

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------

    def _slot_dir(self, slot: str) -> Path:
        return self.dir / slot

    def _engine(self, slot: str) -> _SlotEngine:
        with self._lock:
            return self._engine_locked(slot)

    def _engine_locked(self, slot: str) -> _SlotEngine:
        eng = self._engines.get(slot)
        if eng is None:
            eng = _SlotEngine(
                self.config, provider=self.provider, state_dir=self._slot_dir(slot), slot=slot
            )
            if self.obs.events is not None:
                # Engine events land in the project's log, tagged with the slot.
                eng.events = EventLog(self.obs.events.dir).bind(provider=self.provider.name, slot=slot)
            eng.on_watchdog = self._on_watchdog
            self._engines[slot] = eng
        return eng

    def _save(self, m: PoolMachine) -> None:
        path = self._slot_dir(m.slot) / "pool.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = asdict(m)
            data["billed_at"] = self._billed_at.get(m.slot)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pool-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(data, handle, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _forget(self, m: PoolMachine) -> None:
        with self._lock:
            if self._machines.get(m.slot) is m:
                self._machines.pop(m.slot, None)
                self._billed_at.pop(m.slot, None)
                self._deadline_set.discard(m.slot)
        try:
            (self._slot_dir(m.slot) / "pool.json").unlink()
        except OSError:
            pass

    def _load(self) -> None:
        if not self.dir.is_dir():
            return
        names = {f.name for f in fields(PoolMachine)}
        for path in sorted(self.dir.glob("*/pool.json")):
            try:
                data = json.loads(path.read_text())
                m = PoolMachine(**{k: v for k, v in data.items() if k in names})
            except (OSError, ValueError, TypeError):
                continue
            if m.state == MachineState.RETIRED or m.slot != path.parent.name:
                continue
            if m.state in _WATCHED:
                # Silence before this process started is not evidence: the clock for
                # unresponsive_after starts now.
                m.last_seen_at = max(m.last_seen_at or 0.0, self.clock())
            self._machines[m.slot] = m
            billed = data.get("billed_at")
            self._billed_at[m.slot] = float(billed) if isinstance(billed, (int, float)) else self.clock()

    def machines(self) -> list[PoolMachine]:
        """Every machine the pool tracks, oldest first."""
        with self._lock:
            return sorted(self._machines.values(), key=lambda m: (m.opened_at or 0.0, m.slot))

    # ------------------------------------------------------------------
    # open
    # ------------------------------------------------------------------

    def _free_slot(self) -> str:
        n = 1
        while True:
            slot = f"s{n}"
            held = self._machines.get(slot)
            if held is None or (held.state in (MachineState.GONE, MachineState.RETIRED) and not held.jobs):
                return slot
            n += 1

    def _startup_allowance_s(self) -> float:
        margin = parse_duration(self.queue.defaults.deadline_margin) or 0.0
        return self.config.boot_timeout_s + self.config.startup_timeout_s + margin

    def open(self) -> Optional[PoolMachine]:
        """Start renting a machine and return at once (state provisioning).

        None while open attempts back off after a capacity or provider failure, or
        when the pool already has max_machines; the jobs simply keep waiting."""
        now = self.clock()
        with self._lock:
            if self.backing_off(now):
                return None
            counted = [m for m in self._machines.values() if m.state in _HELD]
            if len(counted) >= self.queue.max_machines:
                return None
            slot = self._free_slot()
            self.last_open_at = now
            m = PoolMachine(
                slot=slot,
                instance_id=None,
                state=MachineState.PROVISIONING,
                hourly_usd=float(self._cached_offer_price() or 0.0),
                stopped_hourly_usd=0.0,
                ready_at=None,
                deadline_at=None,
                opened_at=now,
            )
            self._machines[slot] = m
            self._billed_at[slot] = now
            self._deadline_set.discard(slot)
            self._engines.pop(slot, None)  # a reused slot starts from a fresh Engine
        self._save(m)
        self.obs.emit(ev.POOL_MACHINE_REQUESTED, slot=slot)
        self._start(m, fresh=True)
        return m

    def backing_off(self, now: Optional[float] = None) -> bool:
        """True while open attempts wait after a failure (open() then returns None)."""
        now = self.clock() if now is None else now
        return self.last_failed_at is not None and now - self.last_failed_at < self.open_backoff_s

    def _start(self, m: PoolMachine, fresh: bool) -> None:
        thread = threading.Thread(
            target=self._bring_up, args=(m, fresh), name=f"pool-{m.slot}", daemon=True
        )
        with self._lock:
            self._threads[m.slot] = thread
        thread.start()

    def _bring_up(self, m: PoolMachine, fresh: bool) -> None:
        """Rent (or resume) a machine, give it a deadline, health-check it. Runs in a thread."""
        eng = self._engine(m.slot)
        try:
            try:
                if fresh:
                    state = eng.setup(wait=True)
                else:
                    if not eng.store.load().has_instance():
                        # Never got an instance: nothing to resume (orphans() catches a leak).
                        self._end_thread(m)
                        self._forget(m)
                        return
                    state = eng.ensure_ready()
            except Exception as exc:  # noqa: BLE001 - a thread must report, not die
                self._open_failed(m, eng, exc, fresh)
                return
            with self._lock:
                self._adopt_instance(m, state)
                if fresh:
                    self._open_succeeded()
                m.state = MachineState.CHECKING
            try:
                self._save(m)
                with self._lock:
                    need_deadline = m.slot not in self._deadline_set
                if need_deadline:
                    self._push_deadline(m, eng, state, self.clock() + self._startup_allowance_s())
                result = self.health_check(m)
            except Exception as exc:  # noqa: BLE001 - the machine runs: hand it to the coordinator
                self._crashed(m, exc, "bring_up")
                return
            if not result.ok:
                try:
                    self.retire(m, f"health check failed: {result.detail[-200:]}")
                except Exception as exc:  # noqa: BLE001
                    self._crashed(m, exc, "retire")
                return
            now = self.clock()
            with self._lock:
                self._threads.pop(m.slot, None)
                m.state = MachineState.READY
                m.ready_at = now
                self._resumes.pop(m.slot, None)  # recovered: its restart budget starts over
                m.last_seen_at = now
            self._save(m)
            cold = max(0.0, now - (m.opened_at or now))
            if fresh:
                self.obs.timing("gpu_pool_cold_start_seconds", cold)
            self.obs.emit(
                ev.POOL_MACHINE_READY, slot=m.slot, instance_id=m.instance_id, cold_start_s=round(cold, 1)
            )
            self._gauges()
        finally:
            self._end_thread(m)

    def _crashed(self, m: PoolMachine, exc: BaseException, operation: str) -> None:
        """Something unexpected after the rent: leave the machine unhealthy for retire()."""
        detail = f"{type(exc).__name__}: {str(exc)[:300]}"
        with self._lock:
            m.state = MachineState.UNHEALTHY
        try:
            self._save(m)
        except OSError:
            pass
        self.obs.emit(ev.PROVIDER_ERROR, operation=f"pool_{operation}", slot=m.slot, error_type=type(exc).__name__, error=str(exc)[:300])
        self.obs.emit(ev.POOL_MACHINE_UNHEALTHY, slot=m.slot, instance_id=m.instance_id, detail=detail)

    def _end_thread(self, m: PoolMachine) -> None:
        with self._lock:
            if self._threads.get(m.slot) is threading.current_thread():
                self._threads.pop(m.slot, None)

    def _adopt_instance(self, m: PoolMachine, state: RuntimeState) -> None:
        m.instance_id = state.instance_id
        if state.hourly_cost_usd:
            m.hourly_usd = float(state.hourly_cost_usd)
        inst = state.instance
        if inst is not None and inst.stopped_hourly_cost_usd and not self.provider.capabilities.stop_eliminates_billing:
            m.stopped_hourly_usd = float(inst.stopped_hourly_cost_usd)

    def _open_succeeded(self) -> None:
        self.open_backoff_s = 0.0
        self.last_failed_at = None
        self._blocked = None

    def _open_failed(self, m: PoolMachine, eng: Engine, exc: BaseException, fresh: bool) -> None:
        if fresh:
            now = self.clock()
            with self._lock:
                if not self.backing_off(now):
                    # A second failure inside the same window (opens that ran together)
                    # does not double it again.
                    self.open_backoff_s = (
                        BACKOFF_FIRST_S if not self.open_backoff_s else min(self.open_backoff_s * 2, BACKOFF_MAX_S)
                    )
                    self.last_failed_at = now
            if isinstance(exc, CapacityUnavailableError):
                key, event, fields_ = "capacity", ev.POOL_CAP_BLOCKED, {"cap": "provider_capacity"}
            elif isinstance(exc, BudgetExceededError):
                key, event, fields_ = "budget", ev.POOL_CAP_BLOCKED, {"cap": "max_hourly_cost_usd"}
            else:
                key = f"error:{type(exc).__name__}"
                event, fields_ = ev.PROVIDER_ERROR, {"operation": "pool_open", "error_type": type(exc).__name__}
            with self._lock:
                changed = key != self._blocked
                self._blocked = key
            if changed:
                self.obs.emit(event, error=str(exc)[:500], retry_in_s=self.open_backoff_s, slot=m.slot, **fields_)
        try:
            left = eng.store.load()
        except JambuError:
            left = None
        if left is not None and left.instance_id:
            # Setup could not clean up after itself: keep it so retire() is retried.
            with self._lock:
                self._adopt_instance(m, left)
                m.state = MachineState.UNHEALTHY
            self._save(m)
            self.obs.emit(ev.POOL_MACHINE_UNHEALTHY, slot=m.slot, instance_id=m.instance_id, detail=str(exc)[:500])
            self.retire(m, f"setup failed: {exc}")
            return
        with self._lock:
            m.state = MachineState.GONE
        self._end_thread(m)
        self._forget(m)

    # ------------------------------------------------------------------
    # deadlines and activity
    # ------------------------------------------------------------------

    def _on_watchdog(self, eng: _SlotEngine, state: RuntimeState) -> None:
        with self._lock:
            m = self._machines.get(eng.slot or "")
        if m is None:
            return
        self._push_deadline(m, eng, state, self.clock() + self._startup_allowance_s())

    def _push_deadline(self, m: PoolMachine, eng: Engine, state: RuntimeState, until: float) -> bool:
        client = eng.beacon(state)
        if client is None:
            return False
        until = max(until, m.deadline_at or 0.0)
        try:
            got = float(client.set_deadline(until)["deadline_at"])
        except (WatchdogUnavailable, KeyError, TypeError, ValueError):
            return False
        with self._lock:
            m.deadline_at = got
            self._deadline_set.add(m.slot)
        self._save(m)
        self.obs.emit(ev.POOL_DEADLINE_EXTENDED, slot=m.slot, instance_id=m.instance_id, deadline_at=got)
        self.obs.gauge("gpu_pool_deadline_remaining_seconds", got - self.clock(), slot=m.slot)
        return True

    def extend_deadline(self, m: PoolMachine, until: float) -> None:
        """Push the machine's deadline to ``until`` (never earlier than it is)."""
        eng = self._owned(m)
        got = eng.set_deadline(max(until, m.deadline_at or 0.0))
        with self._lock:
            m.deadline_at = got
            self._deadline_set.add(m.slot)
        self._save(m)
        self.obs.emit(ev.POOL_DEADLINE_EXTENDED, slot=m.slot, instance_id=m.instance_id, deadline_at=got)
        self.obs.gauge("gpu_pool_deadline_remaining_seconds", got - self.clock(), slot=m.slot)

    def _ensure_deadline(self, m: PoolMachine) -> None:
        """An adopted or restarted machine gets its deadline before any work."""
        with self._lock:
            if m.slot in self._deadline_set:
                return
        self.extend_deadline(m, self.clock() + self._startup_allowance_s())

    def keepalive(self, m: PoolMachine) -> None:
        """Count as activity on the machine, so its idle_timeout starts over."""
        client = self._client(m)
        if client.activity() is None:
            raise WatchdogUnavailable(f"machine {m.slot}'s watchdog did not take the keepalive")

    # ------------------------------------------------------------------
    # the machine's watchdog and job runner
    # ------------------------------------------------------------------

    def _owned(self, m: PoolMachine) -> _SlotEngine:
        """The slot's Engine, unless its bring-up thread still owns it."""
        with self._lock:
            thread = self._threads.get(m.slot)
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            raise StateError(f"machine {m.slot} is still starting ({m.state})")
        return self._engine(m.slot)

    def _client(self, m: PoolMachine) -> WatchdogClient:
        eng = self._owned(m)
        client = eng.beacon(eng.store.load())
        if client is None:
            raise StateError(f"machine {m.slot} has no watchdog to reach")
        return client

    def _jobs(self, m: PoolMachine) -> JobClient:
        jobs = JobClient(self._client(m))
        jobs.ensure_runner()
        return jobs

    def health_check(self, m: PoolMachine) -> HealthResult:
        """Run the health check as a job on the machine; a failure marks it unhealthy."""
        hc = self.queue.health_check
        timeout_s = parse_duration(hc.timeout) or 120.0
        command = builtin_health_command(hc.command, hc.min_free_vram_gb)
        job_id = f"{HEALTH_JOB_PREFIX}{m.slot}-{int(time.time() * 1000)}"
        try:
            jobs = self._jobs(m)
            jobs.start(job_id, {"command": command, "timeout_s": timeout_s, "label": "pool health check", "env": {}})
            end = time.time() + timeout_s + HEALTH_SLACK_S
            info: dict = {"state": "running"}
            while time.time() < end:
                info = jobs.info(job_id)
                if info.get("state") != "running":
                    break
                time.sleep(0.25)
            log = ""
            try:
                log = str(jobs.log(job_id, 0).get("data") or "")
            except (StateError, WatchdogUnavailable):
                pass
            if info.get("state") == "running":
                jobs.cancel(job_id)
                result = HealthResult(False, f"health check still running after {timeout_s:g}s")
            elif info.get("state") == "succeeded" and info.get("returncode") == 0:
                result = HealthResult(True, log.strip()[-500:])
            else:
                result = HealthResult(
                    False, f"exit {info.get('returncode')} ({info.get('state')}): {log.strip()[-400:]}"
                )
        except (StateError, WatchdogUnavailable) as exc:
            result = HealthResult(False, f"health check could not run: {exc}")
        if not result.ok:
            with self._lock:
                m.health_failures += 1
                m.state = MachineState.UNHEALTHY
            self._save(m)
            self.obs.inc("gpu_pool_health_failures_total")
            self.obs.emit(ev.POOL_MACHINE_UNHEALTHY, slot=m.slot, instance_id=m.instance_id, detail=result.detail)
        return result

    def dispatch(self, m: PoolMachine, remote_id: str, bundle: Path, manifest: dict, env: dict) -> str:
        """Upload ``bundle`` and start it as ``remote_id``. "already_running" when the
        machine already has that job (a dispatch repeated after a crash): never a second start."""
        self._ensure_deadline(m)
        eng = self._owned(m)
        state = eng.store.load()
        jobs = self._jobs(m)
        command = manifest.get("run") or manifest.get("command")
        if not command:
            raise StateError(f"job {remote_id} has nothing to run (no run: in its manifest)")
        timeout_s = manifest.get("timeout_s")
        if timeout_s is None and manifest.get("timeout"):
            timeout_s = parse_duration(str(manifest["timeout"]))
        merged = {**eng.job_env(state), **dict(manifest.get("env") or {}), **dict(env or {})}
        payload = {
            "command": command,
            "setup": manifest.get("setup"),
            "env": {k: str(v) for k, v in merged.items()},
            "timeout_s": timeout_s,
            "label": str(manifest.get("label") or remote_id),
        }
        try:
            bundle = Path(bundle)
            if bundle.is_file() and bundle.stat().st_size > 0:
                jobs.upload(remote_id, bundle)
            jobs.start(remote_id, payload)
        except JobConflict:
            outcome = "already_running"
        else:
            outcome = "started"
        with self._lock:
            if remote_id not in m.jobs:
                m.jobs.append(remote_id)
            if m.state == MachineState.READY:
                m.state = MachineState.BUSY
        self._save(m)
        return outcome

    def job_info(self, m: PoolMachine, remote_id: str) -> dict:
        return self._jobs(m).info(remote_id)

    def fetch_outputs(self, m: PoolMachine, remote_id: str, dest: Path) -> list[str]:
        return self._jobs(m).download_outputs(remote_id, Path(dest))

    def cancel(self, m: PoolMachine, remote_id: str) -> None:
        self._jobs(m).cancel(remote_id)

    def remote_jobs(self, m: PoolMachine) -> list[dict]:
        """The machine's queue jobs as its watchdog lists them (health checks and model
        servers left out); empty while the machine is still starting."""
        return self.remote_view(m)["jobs"]

    def remote_view(self, m: PoolMachine) -> dict:
        """``remote_jobs`` plus, from the same request, the jobs whose code reached the
        machine but never started (``uploads``: id -> {state, age_s}) and the age of its
        jobs directory (``jobs_dir_age_s``). Both None when the machine cannot say (still
        starting, or a runner from before 0.14.0)."""
        with self._lock:
            thread = self._threads.get(m.slot)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            return {"jobs": [], "uploads": None, "jobs_dir_age_s": None}
        view = self._jobs(m).view()
        view["jobs"] = [
            j for j in view["jobs"]
            if not str(j.get("id", "")).startswith(HEALTH_JOB_PREFIX) and not j.get("service")
        ]
        return view

    # ------------------------------------------------------------------
    # retire
    # ------------------------------------------------------------------

    def retire(self, m: PoolMachine, reason: str, mark_bad: bool = True) -> RetireResult:
        """Pull the outputs the machine still holds, destroy it and mark its host bad
        (never for a machine that is gone: nothing says its host failed).

        The result says which jobs' outputs were saved and which were lost. A failure
        to save outputs never keeps a machine running. A destroy the provider refuses
        leaves the machine listed (unhealthy) to retry, with ``destroyed`` False."""
        eng = self._owned(m)
        if m.state == MachineState.GONE:
            mark_bad = False
        state = eng.store.load()
        host_id = state.instance.host_id if state.instance is not None else None
        result = RetireResult()
        try:
            self._pull_left_outputs(m, eng, state, result)
        except Exception as exc:  # noqa: BLE001 - outputs never keep a machine alive
            self.obs.emit(ev.PROVIDER_ERROR, operation="pool_retire_outputs", slot=m.slot, error_type=type(exc).__name__)
            # Whatever was not pulled before the failure is lost, and says so.
            result.lost.extend(j for j in m.jobs if j not in result.pulled and j not in result.lost)
        now = self.clock()
        self._bill(m, now)
        try:
            eng.destroy(reason=f"pool: {reason}"[:200], discard_outputs=True)
        except JambuError as exc:
            self.obs.emit(
                ev.PROVIDER_ERROR, operation="pool_retire", slot=m.slot, instance_id=m.instance_id, error=str(exc)[:500]
            )
            with self._lock:
                if m.state not in (MachineState.GONE, MachineState.UNRESPONSIVE):
                    m.state = MachineState.UNHEALTHY
            self._save(m)
            return result
        result.destroyed = True
        if host_id and mark_bad:
            BadHosts().mark(self.provider.name, host_id, reason)
        with self._lock:
            m.state = MachineState.RETIRED
            self._resumes.pop(m.slot, None)
        self._forget(m)
        self.obs.emit(
            ev.POOL_MACHINE_RETIRED, slot=m.slot, instance_id=m.instance_id, reason=reason, host_id=host_id,
            host_marked_bad=bool(host_id and mark_bad),
            outputs_pulled=len(result.pulled), outputs_pulled_ids=sorted(result.pulled),
            outputs_lost=len(result.lost), outputs_lost_ids=list(result.lost),
        )
        self._gauges()
        return result

    def _pull_left_outputs(self, m: PoolMachine, eng: Engine, state: RuntimeState, result: RetireResult) -> None:
        """Download every queue job's outputs the machine still holds unsaved; jobs the
        pool knows of (``m.jobs``) count as lost when the machine cannot be reached."""
        client = eng.beacon(state)
        reachable = client is not None and bool(state.instance_id) and client.ping()
        listed: Optional[list[dict]] = None
        if reachable:
            try:
                jobs = JobClient(client)
                jobs.ensure_runner()
                listed = jobs.list()
            except (StateError, WatchdogUnavailable):
                listed = None
        if listed is None:
            result.lost.extend(j for j in m.jobs if j not in result.lost)
            return
        for job in listed:
            rid = str(job.get("id", ""))
            if rid.startswith(HEALTH_JOB_PREFIX) or job.get("service"):
                continue
            if not job.get("outputs_unsaved") and rid not in m.jobs:
                continue
            dest = self._slot_dir(m.slot) / "retired-outputs" / rid
            try:
                jobs.download_outputs(rid, dest)
            except (StateError, WatchdogUnavailable, OSError):
                result.lost.append(rid)
                continue
            result.pulled[rid] = dest
        seen = {str(j.get("id", "")) for j in listed}
        result.lost.extend(j for j in m.jobs if j not in seen and j not in result.lost)

    def retire_instance(self, instance: Instance, reason: str) -> None:
        """Destroy a provider instance no slot tracks (an orphan, see ``orphans()``)."""
        self.provider.destroy(instance)
        self.obs.emit(ev.POOL_MACHINE_RETIRED, instance_id=instance.id, reason=reason, orphan=True)

    # ------------------------------------------------------------------
    # poll
    # ------------------------------------------------------------------

    def poll(self) -> list[PoolMachine]:
        """Reconcile every machine with the provider and its watchdog; bill what ran."""
        now = self.clock()
        for m in self.machines():
            self._bill(m, now)
            with self._lock:
                thread = self._threads.get(m.slot)
            if thread is not None and thread.is_alive():
                continue
            if m.state in _BOOTING:
                # The thread that was bringing it up is gone (a restarted coordinator).
                self._resume(m)
                continue
            if m.state in (MachineState.GONE, MachineState.RETIRED, MachineState.UNRESPONSIVE):
                continue
            try:
                self._reconcile(m, now)
            except Exception as exc:  # noqa: BLE001 - one machine never stops the poll
                self.obs.emit(ev.PROVIDER_ERROR, operation="pool_poll", slot=m.slot, error=str(exc)[:500])
        self._gauges()
        return self.machines()

    def _resume(self, m: PoolMachine) -> None:
        with self._lock:
            tries = self._resumes.get(m.slot, 0) + 1
            self._resumes[m.slot] = tries
        if tries > MAX_RESUMES:
            with self._lock:
                m.state = MachineState.UNHEALTHY
            self._save(m)
            self.obs.emit(
                ev.POOL_MACHINE_UNHEALTHY, slot=m.slot, instance_id=m.instance_id,
                detail=f"bring-up did not finish after {MAX_RESUMES} attempts",
            )
            return
        self._start(m, fresh=False)

    def _reconcile(self, m: PoolMachine, now: float) -> None:
        eng = self._engine(m.slot)
        state = eng.store.load()
        if not state.instance_id:
            self._mark_gone(m, eng, "no instance on record")
            return
        instance = state.instance or Instance(id=state.instance_id, provider=self.provider.name)
        try:
            status = self.provider.status(instance)
        except ProviderError as exc:
            self.obs.emit(ev.PROVIDER_ERROR, operation="status", slot=m.slot, error=str(exc)[:500])
            return
        if not status.exists:
            self._mark_gone(m, eng, "instance missing at provider")
            return
        if status.state is InstanceState.STOPPED:
            if m.state != MachineState.STOPPED:
                with self._lock:
                    m.state = MachineState.STOPPED
                    self._deadline_set.discard(m.slot)
                self._save(m)
            return
        if status.state is InstanceState.FAILED:
            if m.state != MachineState.UNHEALTHY:
                with self._lock:
                    m.state = MachineState.UNHEALTHY
                self._save(m)
                self.obs.emit(
                    ev.POOL_MACHINE_UNHEALTHY, slot=m.slot, instance_id=m.instance_id,
                    detail=f"provider reports failed: {status.message or ''}"[:500],
                )
            return
        running = status.state in (InstanceState.READY, InstanceState.BUSY, InstanceState.IDLE)
        if not running:
            return
        if m.state == MachineState.STOPPED:
            # Restarted (outside the pool, or by a wake for outputs): deadline and health first.
            with self._lock:
                m.state = MachineState.CHECKING
                self._deadline_set.discard(m.slot)
            self._save(m)
            self._resume(m)
            return
        if m.state not in _WATCHED:
            return
        client = eng.beacon(state)
        remote = client.try_state() if client is not None else None
        if remote is None:
            since = m.last_seen_at or m.ready_at or m.opened_at or now
            if now - since >= self.queue.unresponsive_after_s:
                with self._lock:
                    m.state = MachineState.UNRESPONSIVE
                self._save(m)
                self.obs.emit(
                    ev.POOL_MACHINE_UNRESPONSIVE, slot=m.slot, instance_id=m.instance_id,
                    silent_s=round(now - since, 1),
                )
            return
        with self._lock:
            m.last_seen_at = now
        if remote.get("deadline_at") is None or (m.deadline_at and float(remote["deadline_at"]) < m.deadline_at - 1):
            # The watchdog lost its deadline (restart) or never had one from us (adopted).
            with self._lock:
                self._deadline_set.discard(m.slot)
            self._push_deadline(m, eng, state, max(m.deadline_at or 0.0, now + self._startup_allowance_s()))
        elif m.deadline_at:
            with self._lock:
                self._deadline_set.add(m.slot)
        try:
            listed = self.remote_jobs(m)
        except (StateError, WatchdogUnavailable):
            listed = None
        with self._lock:
            if listed is not None:
                # Running jobs, and finished ones whose outputs exist only on the machine.
                m.jobs = [
                    str(j["id"]) for j in listed
                    if j.get("state") == "running" or j.get("outputs_unsaved")
                ]
                running = any(j.get("state") == "running" for j in listed)
                if m.state in (MachineState.READY, MachineState.BUSY):
                    m.state = MachineState.BUSY if running else MachineState.READY
        self._save(m)

    def _mark_gone(self, m: PoolMachine, eng: Engine, why: str) -> None:
        with eng.store.process_lock():
            state = eng.store.load()
            if state.instance_id:
                state.clear_instance(InstanceState.DESTROYED.value)
                eng.store.save(state)
        with self._lock:
            m.state = MachineState.GONE
        self._save(m)
        self.obs.emit(ev.POOL_MACHINE_GONE, slot=m.slot, instance_id=m.instance_id, reason=why)

    # ------------------------------------------------------------------
    # money and metrics
    # ------------------------------------------------------------------

    def _bill(self, m: PoolMachine, now: float) -> None:
        with self._lock:
            since = self._billed_at.get(m.slot, m.opened_at or now)
            rate = machine_usd_per_hour(m)
            if rate <= 0 and m.state in _BOOTING:
                return  # price not known yet: the boot is billed once it is
            jobs = list(m.jobs)
            self._billed_at[m.slot] = max(since, now)
        if rate > 0 and now > since:
            self.spend.record(m.slot, since, now, rate, jobs=jobs)
            if m.state != MachineState.RETIRED:
                self._save(m)  # billed_at on disk: a restarted pool never bills this again

    def _gauges(self) -> None:
        machines = self.machines()
        for state in MachineState.ALL:
            self.obs.gauge("gpu_pool_machines", sum(1 for m in machines if m.state == state), state=state)
        self.obs.gauge("gpu_pool_usd_per_hour", pool_usd_per_hour(machines))
        self.obs.gauge("gpu_pool_spend_usd_total", self.spend.total())
        now = self.clock()
        for m in machines:
            if m.deadline_at:
                self.obs.gauge("gpu_pool_deadline_remaining_seconds", m.deadline_at - now, slot=m.slot)

    # ------------------------------------------------------------------
    # what the pool would rent next
    # ------------------------------------------------------------------

    def _next_offer(self) -> Optional[Any]:
        now = self.clock()
        if self._offer_at is not None and now - self._offer_at < OFFER_CACHE_S:
            return self._offer
        spec = self._probe.build_spec()
        spec.exclude_hosts = BadHosts().recent(self.provider.name)
        try:
            offers = self.provider.find_offers(spec, limit=1)
        except JambuError:
            offers = []
        self._offer = offers[0] if offers else None
        self._offer_at = now
        return self._offer

    def _cached_offer_price(self) -> Optional[float]:
        offer = self._offer if self._offer_at is not None else None
        return float(offer.hourly_cost_usd) if offer is not None else None

    def next_offer_usd(self) -> Optional[float]:
        """The hourly price of the machine the next open() would rent; None when none is on offer."""
        offer = self._next_offer()
        return float(offer.hourly_cost_usd) if offer is not None else None

    def card_vram_gb(self) -> Optional[float]:
        """The VRAM of the pool's card, from the offer it rents from; None when unknown."""
        offer = self._next_offer()
        return float(offer.gpu_vram_gb) if offer is not None and offer.gpu_vram_gb else None

    # ------------------------------------------------------------------
    # orphans
    # ------------------------------------------------------------------

    def label_prefix(self) -> str:
        return f"{JAMBU_LABEL_PREFIX}{self._probe._label_name()[:20]}-"

    def orphans(self) -> list[Instance]:
        """Provider instances labeled as this project's pool machines (a slot, this
        config's fingerprint) that no slot tracks."""
        if not self.provider.capabilities.list_instances:
            return []
        prefix = self.label_prefix()
        suffix = f"-{self.config.fingerprint()[:6]}"
        tracked: set[str] = set()
        with self._lock:
            slots = list(self._machines.values())
            # A slot still coming up may have rented an instance it has not recorded yet.
            booting = {
                m.slot for m in slots
                if m.state in _BOOTING or (self._threads.get(m.slot) is not None and self._threads[m.slot].is_alive())
            }
        for m in slots:
            if m.instance_id:
                tracked.add(m.instance_id)
            try:
                held = self._engine(m.slot).store.load().instance_id
            except JambuError:
                held = None
            if held:
                tracked.add(held)
        now = self.clock()
        seen = self._load_seen()
        current: dict[str, float] = {}
        out = []
        for inst in self.provider.list_instances():
            label = inst.label or ""
            if not (label.startswith(prefix) and label.endswith(suffix)):
                continue
            slot = label[len(prefix): -len(suffix)]
            if not _SLOT.fullmatch(slot) or slot in booting or inst.id in tracked:
                continue
            # Some providers do not say when an instance was created (their mappers
            # fill in "now"): age it from when this pool first saw it too.
            first = current[inst.id] = seen.get(inst.id, now)
            born = min(float(inst.created_at), first) if inst.created_at else first
            if now - born < ORPHAN_MIN_AGE_S:
                continue
            out.append(inst)
        self._save_seen(current)
        return out

    def _seen_path(self) -> Path:
        return self.dir / "seen.json"

    def _load_seen(self) -> dict[str, float]:
        with self._lock:
            if self._seen is None:
                try:
                    data = json.loads(self._seen_path().read_text())
                    self._seen = {str(k): float(v) for k, v in data.items()} if isinstance(data, dict) else {}
                except (OSError, ValueError, TypeError):
                    self._seen = {}
            return dict(self._seen)

    def _save_seen(self, current: dict[str, float]) -> None:
        """Keep first sightings of the untracked instances still listed; drop the rest."""
        with self._lock:
            if current == self._seen:
                return
            self._seen = dict(current)
        path = self._seen_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".seen-", suffix=".json")
            with os.fdopen(fd, "w") as handle:
                json.dump(current, handle)
            os.replace(tmp, path)
        except OSError:
            pass
