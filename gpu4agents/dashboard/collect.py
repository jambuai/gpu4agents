"""Host-wide, read-only snapshot collectors for the dashboard.

Every collector reads local state only (the sweep registry and each project's
``.gpu/state.json``); none reaches a provider API or returns a secret. A
missing or corrupt state file yields an ``error`` row, never an exception, so
one broken project never blanks the page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..core.state import StateStore


def select_instances(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The GPU-centric view the page renders: rows that have an instance, plus
    error rows (a project whose state could not be read still deserves a cell).
    A project with no instance and no error (destroyed) is not a GPU to show.
    """
    return [r for r in projects if r.get("instance_id") or r.get("error")]


def project_snapshots(config_paths: Iterable[Path], now: float) -> list[dict[str, Any]]:
    """One read-only row per project config path (a ``gpu.yaml``).

    ``config_paths`` are what ``ProjectRegistry().paths()`` returns. The row
    never carries ``api_key`` or any other secret; ``reachable`` is left None
    for the reachability watch to fill in.
    """
    rows: list[dict[str, Any]] = []
    for raw in config_paths:
        cfg = Path(raw)
        project_dir = cfg.parent
        row: dict[str, Any] = {"project": project_dir.name, "dir": str(project_dir)}
        try:
            state = StateStore(project_dir / ".gpu").load()
        except Exception as exc:  # noqa: BLE001 - a broken project is a row, not a crash
            row["error"] = str(exc)
            rows.append(row)
            continue
        idle = None if state.last_activity is None else now - state.last_activity
        row.update(
            {
                "instance_id": state.instance_id,
                "model": state.model_id,
                "endpoint": state.endpoint,
                "state": state.state,
                "hourly_cost_usd": state.hourly_cost_usd,
                "session_cost_usd": state.session_cost_usd,
                "idle_seconds": idle,
                "reachable": None,
            }
        )
        rows.append(row)
    return rows
