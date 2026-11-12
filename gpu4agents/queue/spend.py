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
    each pays an equal share. An interval with no jobs counts in the total only.

    Loaded once on construction; totals are kept in memory and updated by record()."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._total = 0.0
        self._by_job: dict[str, float] = {}
        for row in self._rows():
            self._add(row)

    def _add(self, row: dict) -> None:
        self._total += row["usd"]
        jobs = row["jobs"]
        for j in jobs:
            self._by_job[j] = self._by_job.get(j, 0.0) + row["usd"] / len(jobs)

    def record(
        self, slot: str, from_ts: float, to_ts: float, usd_per_hour: float, jobs: Optional[list[str]] = None
    ) -> float:
        usd = max(0.0, to_ts - from_ts) * usd_per_hour / 3600
        row = {"slot": slot, "from": from_ts, "to": to_ts, "usd_per_hour": usd_per_hour, "usd": usd, "jobs": list(jobs or [])}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+b") as f:
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
            else:
                torn = False
            f.seek(0, os.SEEK_END)
            f.write((b"\n" if torn else b"") + (json.dumps(row) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        self._add(row)
        return usd

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
                usd = r["usd"]
                jobs = r.get("jobs") or []
                if isinstance(usd, bool) or not isinstance(usd, (int, float)):
                    continue
                if not isinstance(jobs, list) or not all(isinstance(j, str) for j in jobs):
                    continue
            except (ValueError, KeyError, TypeError, AttributeError):
                continue  # torn or malformed line
            rows.append({"usd": float(usd), "jobs": jobs})
        return rows

    def total(self) -> float:
        return self._total

    def by_job(self) -> dict[str, float]:
        return dict(self._by_job)
