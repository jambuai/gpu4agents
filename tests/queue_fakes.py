"""A Pool in memory, with scripted job outcomes, for coordinator and auditor tests.

It has the public interface of ``gpu4agents.queue.pool.Pool``: machines open, become
ready on a later ``poll()``, run what ``dispatch`` gives them and finish each job
after a scripted number of polls with a scripted exit code and outputs. Tests make
machines vanish (``lose``), stop answering (``hang``) or fail health checks.

A dispatched job is unpacked like the watchdog does (bundle into
``<root>/<remote id>/src``), so a test can look at what the job would see.
"""

from __future__ import annotations

import itertools
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from gpu4agents.core import events as ev
from gpu4agents.core.beacon import WatchdogUnavailable
from gpu4agents.core.errors import StateError
from gpu4agents.core.models import Instance
from gpu4agents.queue.observe import Observer
from gpu4agents.queue.pool import HealthResult, RetireResult
from gpu4agents.queue.pool_types import MachineState, PoolMachine

_LIVE = (MachineState.READY, MachineState.BUSY)
_INSTANCE_IDS = itertools.count(1)  # unique across FakePools, like provider instance ids


@dataclass
class FakeRun:
    """How one dispatch of a job goes on the fake machine."""

    exit_code: Optional[int] = 0
    outputs: Optional[dict[str, bytes]] = None  # None -> {"result.txt": b"done <remote id>"}
    polls: int = 1  # polls until it finishes
    state: Optional[str] = None  # final watchdog state override ("timeout", "lost")


@dataclass
class FakeRemote:
    id: str
    slot: str
    spec: FakeRun
    manifest: dict
    env: dict
    workdir: Path
    started_at: float
    state: str = "running"
    returncode: Optional[int] = None
    polls_left: int = 1
    outputs_saved: bool = False

    @property
    def job_id(self) -> str:
        return self.id.rsplit("-a", 1)[0]

    def outputs(self) -> dict[str, bytes]:
        if self.spec.outputs is None:
            return {"result.txt": f"done {self.id}".encode()}
        return dict(self.spec.outputs)

    def record(self) -> dict:
        unsaved = bool(self.outputs()) and (self.state == "running" or not self.outputs_saved)
        return {
            "id": self.id,
            "state": self.state,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "outputs_unsaved": unsaved,
        }


class FakePool:
    def __init__(
        self,
        observer: Observer,
        clock: Callable[[], float],
        root: Path,
        *,
        hourly_usd: float = 1.0,
        vram_gb: Optional[float] = 24.0,
        max_machines: int = 3,
        boot_polls: int = 1,
    ) -> None:
        self.obs = observer
        self.clock = clock
        self.root = Path(root)
        self.hourly_usd = hourly_usd
        self.vram_gb = vram_gb
        self.max_machines = max_machines
        self.boot_polls = boot_polls

        self._machines: dict[str, PoolMachine] = {}
        self._boot_left: dict[str, int] = {}
        self.runs: list[FakeRemote] = []  # every remote job, on any machine
        self.unreachable: set[str] = set()
        # Provider instances labeled as this pool's machines that no slot tracks.
        self.orphan_instances: list[Instance] = []
        self.retired_instances: list[tuple[str, str]] = []  # (instance id, reason)

        self.last_open_at: Optional[float] = None
        self.last_failed_at: Optional[float] = None
        self.open_backoff_s: float = 0.0

        # scripts
        self.scripts: dict[str, list[FakeRun]] = {}  # queue job id -> one FakeRun per dispatch
        self.fetch_failures: dict[str, int] = {}  # remote id -> fetches that fail first
        self.health_failures: list[str] = []  # slots whose next health check fails
        self.retire_refused: set[str] = set()
        self.retire_loses: set[str] = set()  # retire cannot pull anything from these slots
        self.dispatch_failures: dict[str, int] = {}  # slot -> dispatches that fail before starting
        # Slow calls: operation name -> seconds the call takes on the injected clock. The
        # clock moves in steps of ``slow_step_s``; ``during()`` runs after each step (a
        # heartbeat thread, or another coordinator acting meanwhile).
        self.delays: dict[str, float] = {}
        self.slow_step_s = 5.0
        self.during: Optional[Callable[[], None]] = None
        self.advance: Optional[Callable[[float], None]] = None  # how to move the clock

        # what happened
        self.opens = 0
        self.starts: list[FakeRemote] = []
        self.dispatches: list[tuple[str, str, str]] = []  # (slot, remote id, outcome)
        self.deadlines: list[tuple[str, float, float]] = []  # (slot, until, now)
        self.keepalives: list[str] = []
        self.health_checks: list[str] = []
        self.fetches: list[str] = []
        self.cancels: list[str] = []
        self.retired: list[tuple[str, str, bool]] = []  # (slot, reason, mark_bad)

    # ------------------------------------------------------------------ scripting

    def script(self, job_id: str, *runs: FakeRun) -> None:
        self.scripts.setdefault(job_id, []).extend(runs)

    def lose(self, slot: str) -> None:
        """The provider no longer has the machine."""
        self._machines[slot].state = MachineState.GONE
        self.unreachable.add(slot)

    def hang(self, slot: str) -> None:
        """The machine runs but its watchdog stopped answering."""
        self._machines[slot].state = MachineState.UNRESPONSIVE
        self.unreachable.add(slot)

    @property
    def remotes(self) -> dict[str, FakeRemote]:
        """Remote id -> its latest run (a test convenience)."""
        return {r.id: r for r in self.runs}

    def slot_of(self, remote_id: str) -> str:
        return self.remotes[remote_id].slot

    def _slow(self, op: str) -> None:
        left = self.delays.get(op, 0.0)
        while left > 0 and self.advance is not None:
            step = min(self.slow_step_s, left)
            self.advance(step)
            left -= step
            if self.during is not None:
                self.during()

    # ------------------------------------------------------------------ Pool interface

    def machines(self) -> list[PoolMachine]:
        return sorted(self._machines.values(), key=lambda m: (m.opened_at or 0.0, m.slot))

    def backing_off(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return self.last_failed_at is not None and now - self.last_failed_at < self.open_backoff_s

    def open(self) -> Optional[PoolMachine]:
        if self.backing_off():
            return None
        held = [m for m in self._machines.values() if m.state not in (MachineState.GONE, MachineState.RETIRED)]
        if len(held) >= self.max_machines:
            return None
        # Slot names are never reused here, so a test can tell machines apart by slot.
        self.opens += 1
        slot = f"s{self.opens}"
        now = self.clock()
        instance = f"i-{next(_INSTANCE_IDS)}"
        self.last_open_at = now
        m = PoolMachine(
            slot=slot, instance_id=instance, state=MachineState.PROVISIONING,
            hourly_usd=self.hourly_usd, stopped_hourly_usd=0.0, ready_at=None, deadline_at=None,
            opened_at=now + self.opens * 1e-6,
        )
        self._machines[slot] = m
        self._boot_left[slot] = self.boot_polls
        self.obs.emit(ev.POOL_MACHINE_REQUESTED, slot=slot)
        return m

    def poll(self) -> list[PoolMachine]:
        self._slow("poll")
        now = self.clock()
        for m in list(self._machines.values()):
            if m.state == MachineState.PROVISIONING:
                self._boot_left[m.slot] -= 1
                if self._boot_left[m.slot] <= 0:
                    m.state = MachineState.READY
                    m.ready_at = now
                    self.obs.emit(ev.POOL_MACHINE_READY, slot=m.slot)
                continue
            if m.state not in _LIVE:
                continue
            for r in self._on(m.slot):
                if r.state != "running":
                    continue
                r.polls_left -= 1
                if r.polls_left <= 0:
                    r.returncode = r.spec.exit_code
                    r.state = r.spec.state or ("succeeded" if r.spec.exit_code == 0 else "failed")
            self._sync(m)
        return self.machines()

    def health_check(self, m: PoolMachine) -> HealthResult:
        self.health_checks.append(m.slot)
        if m.slot in self.health_failures:
            self.health_failures.remove(m.slot)
            m.health_failures += 1
            m.state = MachineState.UNHEALTHY
            self.obs.emit(ev.POOL_MACHINE_UNHEALTHY, slot=m.slot, detail="scripted")
            return HealthResult(False, "scripted health failure")
        return HealthResult(True, "ok")

    def extend_deadline(self, m: PoolMachine, until: float) -> None:
        self._reach(m)
        got = max(until, m.deadline_at or 0.0)
        m.deadline_at = got
        self.deadlines.append((m.slot, until, self.clock()))
        self.obs.emit(ev.POOL_DEADLINE_EXTENDED, slot=m.slot, deadline_at=got)

    def keepalive(self, m: PoolMachine) -> None:
        self._reach(m)
        self.keepalives.append(m.slot)

    def dispatch(self, m: PoolMachine, remote_id: str, bundle: Path, manifest: dict, env: dict) -> str:
        self._reach(m)
        self._slow("dispatch")
        if self.dispatch_failures.get(m.slot, 0) > 0:
            self.dispatch_failures[m.slot] -= 1
            self.dispatches.append((m.slot, remote_id, "error"))
            raise WatchdogUnavailable(f"upload to {m.slot} cut short (scripted)")
        if any(r.id == remote_id and r.slot == m.slot for r in self.runs):
            # like the job runner's 409: only this machine knows the id
            self.dispatches.append((m.slot, remote_id, "already_running"))
            return "already_running"
        runs = self.scripts.get(remote_id.rsplit("-a", 1)[0]) or []
        spec = runs.pop(0) if runs else FakeRun()
        workdir = self.root / remote_id
        src = workdir / "src"
        src.mkdir(parents=True, exist_ok=True)
        if Path(bundle).is_file():
            with tarfile.open(bundle, "r:*") as tar:
                tar.extractall(src, filter="data")
        r = FakeRemote(
            id=remote_id, slot=m.slot, spec=spec, manifest=dict(manifest), env=dict(env),
            workdir=workdir, started_at=self.clock(), polls_left=max(1, spec.polls),
        )
        self.runs.append(r)
        self.starts.append(r)
        self.dispatches.append((m.slot, remote_id, "started"))
        self._sync(m)
        return "started"

    def _find(self, m: PoolMachine, remote_id: str) -> FakeRemote:
        r = next((r for r in self.runs if r.id == remote_id and r.slot == m.slot), None)
        if r is None:
            raise StateError(f"no job {remote_id}")
        return r

    def job_info(self, m: PoolMachine, remote_id: str) -> dict:
        self._reach(m)
        return self._find(m, remote_id).record()

    def fetch_outputs(self, m: PoolMachine, remote_id: str, dest: Path) -> list[str]:
        self._reach(m)
        self._slow("fetch_outputs")
        self.fetches.append(remote_id)
        r = self._find(m, remote_id)
        if self.fetch_failures.get(remote_id, 0) > 0:
            self.fetch_failures[remote_id] -= 1
            raise WatchdogUnavailable(f"download of {remote_id} cut short (scripted)")
        names = self._write(r, Path(dest))
        if r.state != "running":
            r.outputs_saved = True
        self._sync(m)
        return names

    def cancel(self, m: PoolMachine, remote_id: str) -> None:
        self._reach(m)
        self.cancels.append(remote_id)
        r = next((r for r in self.runs if r.id == remote_id and r.slot == m.slot), None)
        if r is not None and r.state == "running":
            r.state = "cancelled"
            r.returncode = -15
        self._sync(m)

    def remote_jobs(self, m: PoolMachine) -> list[dict]:
        self._reach(m)
        return [r.record() for r in self._on(m.slot)]

    def retire(self, m: PoolMachine, reason: str, mark_bad: bool = True) -> RetireResult:
        if m.state == MachineState.GONE:
            mark_bad = False
        result = RetireResult()
        for r in self._on(m.slot):
            if not r.record()["outputs_unsaved"] and r.id not in m.jobs:
                continue
            if m.slot in self.unreachable or m.slot in self.retire_loses:
                result.lost.append(r.id)
                continue
            dest = self.root / "retired" / r.id
            self._write(r, dest)
            result.pulled[r.id] = dest
        if m.slot in self.retire_refused:
            m.state = MachineState.UNHEALTHY
            return result
        result.destroyed = True
        self.retired.append((m.slot, reason, mark_bad))
        for r in self._on(m.slot):
            if r.state == "running":
                r.state = "lost"
            r.slot = f"{m.slot}-retired"
        m.state = MachineState.RETIRED
        self._machines.pop(m.slot, None)
        self.obs.emit(ev.POOL_MACHINE_RETIRED, slot=m.slot, reason=reason, host_marked_bad=mark_bad)
        return result

    def orphans(self) -> list[Instance]:
        return list(self.orphan_instances)

    def retire_instance(self, instance: Instance, reason: str) -> None:
        self.orphan_instances = [i for i in self.orphan_instances if i.id != instance.id]
        self.retired_instances.append((instance.id, reason))
        self.obs.emit(ev.POOL_MACHINE_RETIRED, instance_id=instance.id, reason=reason, orphan=True)

    def next_offer_usd(self) -> Optional[float]:
        return self.hourly_usd

    def card_vram_gb(self) -> Optional[float]:
        return self.vram_gb

    # ------------------------------------------------------------------ internals

    def _on(self, slot: str) -> list[FakeRemote]:
        return [r for r in self.runs if r.slot == slot]

    def _reach(self, m: PoolMachine) -> None:
        if m.slot in self.unreachable or m.slot not in self._machines:
            raise WatchdogUnavailable(f"machine {m.slot} does not answer")

    def _sync(self, m: PoolMachine) -> None:
        mine = self._on(m.slot)
        m.jobs = [r.id for r in mine if r.record()["outputs_unsaved"]]
        if m.state in _LIVE:
            m.state = MachineState.BUSY if any(r.state == "running" for r in mine) else MachineState.READY

    @staticmethod
    def _write(r: FakeRemote, dest: Path) -> list[str]:
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for name, data in r.outputs().items():
            p = dest / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return sorted(r.outputs())
