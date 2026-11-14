"""The reachability watch: periodically probe each known endpoint and keep a
short rolling window per endpoint, so a transient fault (a slow connect, a
5xx burst, an ``Errno 110`` connection timeout) is visible the moment it
happens rather than reconstructed after the fact.

Probing uses ``httpx`` (already a dependency); the window is in memory only.
Tests drive ``record`` directly, so the background thread stays thin.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import httpx

# (ok, connect_ms, err)
ProbeResult = tuple[bool, Optional[float], Optional[str]]
_TIMEOUT = httpx.Timeout(8.0, connect=5.0)


class ReachabilityWatch:
    def __init__(self, interval_s: float, history_s: float, client: Any = None) -> None:
        self.interval_s = interval_s
        self.history_s = history_s
        self._client = client  # an httpx.Client, or None for module-level httpx
        self._windows: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def record(self, endpoint: str, result: ProbeResult, ts: float) -> None:
        ok, connect_ms, err = result
        with self._lock:
            dq = self._windows.setdefault(endpoint, deque())
            dq.append((ts, ok, connect_ms, err))
            cutoff = ts - self.history_s
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def snapshot(self, endpoint: str) -> dict[str, Any]:
        with self._lock:
            dq = list(self._windows.get(endpoint, ()))
        window = [{"ts": t, "ok": ok, "connect_ms": cm, "err": e} for (t, ok, cm, e) in dq]
        failures = sum(1 for (_, ok, _, _) in dq if not ok)
        errno110 = sum(1 for (_, ok, _, e) in dq if not ok and e and "Errno 110" in e)
        last = dq[-1] if dq else None
        return {
            "last_ok": None if last is None else last[1],
            "last_connect_ms": None if last is None else last[2],
            "window": window,
            "failures_in_window": failures,
            "errno110_in_window": errno110,
        }

    def probe_once(self, endpoint: str) -> ProbeResult:
        client = self._client or httpx
        base = endpoint.rstrip("/")
        t0 = time.monotonic()
        try:
            resp = client.get(base + "/health", timeout=_TIMEOUT)
            ms = (time.monotonic() - t0) * 1000.0
            ok = resp.status_code < 500
            return (ok, ms, None if ok else f"HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001 - try a plain HEAD before giving up
            try:
                resp = client.request("HEAD", base + "/", timeout=_TIMEOUT)
                ms = (time.monotonic() - t0) * 1000.0
                return (resp.status_code < 500, ms, None)
            except Exception as exc2:  # noqa: BLE001
                return (False, None, str(exc2) or str(exc))

    def start(self, endpoints_provider: Callable[[], list[str]]) -> None:
        """Run the probe loop in a daemon thread until ``stop()``."""
        def loop() -> None:
            while not self._stop.is_set():
                for endpoint in endpoints_provider() or []:
                    if endpoint:
                        self.record(endpoint, self.probe_once(endpoint), ts=time.time())
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
