"""Pool spend: what machines cost per hour and a ledger that attributes cost to jobs.

Billing rate by MachineState:
  full rate (hourly_usd): provisioning, checking, ready, busy, draining, unhealthy,
      unresponsive (still rented until destroyed) and held (kept running for outputs).
  disk only (stopped_hourly_usd): stopped.
  nothing: gone, retired (destroyed).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .pool_types import MachineState, PoolMachine

_FREE = (MachineState.GONE, MachineState.RETIRED)


def machine_usd_per_hour(m: PoolMachine) -> float:
    if m.state in _FREE:
        return 0.0
    if m.state == MachineState.STOPPED:
        return m.stopped_hourly_usd
    return m.hourly_usd


def pool_usd_per_hour(machines: list[PoolMachine]) -> float:
    return sum(machine_usd_per_hour(m) for m in machines)


class SpendLedger:
    """Append-only JSONL. ``jobs`` are the jobs that ran on the machine during the interval;
    each pays an equal share. An interval with no jobs counts in the total only."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def record(
        self, slot: str, from_ts: float, to_ts: float, usd_per_hour: float, jobs: Optional[list[str]] = None
    ) -> float:
        usd = max(0.0, to_ts - from_ts) * usd_per_hour / 3600
        row = {"slot": slot, "from": from_ts, "to": to_ts, "usd_per_hour": usd_per_hour, "usd": usd, "jobs": list(jobs or [])}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return usd

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a torn last line from a crash
        return rows

    def total(self) -> float:
        return sum(r["usd"] for r in self._rows())

    def by_job(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self._rows():
            jobs = r.get("jobs") or []
            for j in jobs:
                out[j] = out.get(j, 0.0) + r["usd"] / len(jobs)
        return out
