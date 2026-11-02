"""Machines that recently failed us, remembered across setups.

A host that could not pull the image or reach the network fails the next
setup the same way, and the cheapest-offer search keeps picking it. Each
failure is kept for ``BAD_HOST_TTL_S`` in ``$GPU_HOME/bad_hosts.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .sweep import jambu_home

BAD_HOST_TTL_S = 24 * 3600.0


class BadHosts:
    def __init__(self, path: Optional[Path] = None, ttl_s: float = BAD_HOST_TTL_S) -> None:
        self.path = Path(path) if path else jambu_home() / "bad_hosts.json"
        self.ttl_s = ttl_s

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text() or "{}")
        except (OSError, ValueError):
            # Unreadable memory only costs a retry; never block a setup on it.
            return {}
        return data if isinstance(data, dict) else {}

    def recent(self, provider: str, now: Optional[float] = None) -> list[str]:
        now = now if now is not None else time.time()
        hosts = self._load().get(provider) or {}
        return sorted(
            host
            for host, entry in hosts.items()
            if isinstance(entry, dict) and now - float(entry.get("at") or 0) < self.ttl_s
        )

    def mark(self, provider: str, host_id: str, reason: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        data = self._load()
        hosts = data.setdefault(provider, {})
        hosts[str(host_id)] = {"at": now, "reason": reason[:200]}
        # Drop expired entries so the file stays small.
        data[provider] = {
            h: e for h, e in hosts.items()
            if isinstance(e, dict) and now - float(e.get("at") or 0) < self.ttl_s
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.path)
        except OSError:
            pass
