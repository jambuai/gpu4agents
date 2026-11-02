"""Client for the remote lifecycle watchdog.

The watchdog is the authority on activity: it holds the heartbeat state and
performs the shutdown. The CLI only reports activity to it and reads it back.
Every call is best-effort - losing the watchdog must never break a workload,
and an unreachable watchdog is surfaced as a warning, not a crash.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

import httpx

from .errors import JambuError


class WatchdogUnavailable(JambuError):
    """The remote watchdog could not be reached."""


class WatchdogClient:
    def __init__(self, base_url: str, token: Optional[str], timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Jambu-Token": self.token} if self.token else {}

    # -- transport ----------------------------------------------------------

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = httpx.request(
                method,
                url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise WatchdogUnavailable(f"watchdog unreachable at {url}: {exc}") from exc
        if response.status_code == 401:
            raise WatchdogUnavailable("watchdog rejected the control token")
        if response.status_code >= 400:
            raise WatchdogUnavailable(
                f"watchdog returned {response.status_code}: {response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError:
            return {}

    def try_request(self, method: str, path: str, payload: Optional[dict] = None) -> Optional[dict]:
        try:
            return self._request(method, path, payload)
        except WatchdogUnavailable:
            return None

    # -- routes -------------------------------------------------------------

    def ping(self) -> bool:
        try:
            self._request("GET", "/health")
            return True
        except WatchdogUnavailable:
            return False

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/state")

    def try_state(self) -> Optional[dict[str, Any]]:
        return self.try_request("GET", "/state")

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        data = self.try_request("GET", f"/events?limit={limit}") or {}
        return data.get("events", [])

    def logs(self, name: str = "runtime", lines: int = 200) -> list[str]:
        """Tail of an instance log (runtime | onstart | watchdog).

        Raises WatchdogUnavailable, so a caller can tell "empty" from "unreachable".
        """
        data = self._request("GET", f"/logs?name={quote(name)}&lines={int(lines)}")
        return [str(line) for line in data.get("lines") or []]

    def identify(
        self, instance_id: str, hourly_cost_usd: Optional[float] = None
    ) -> Optional[dict]:
        return self.try_request(
            "POST",
            "/identify",
            {"instance_id": instance_id, "hourly_cost_usd": hourly_cost_usd},
        )

    def activity(self) -> Optional[dict]:
        return self.try_request("POST", "/activity")

    def mark_starting(self) -> Optional[dict]:
        return self.try_request("POST", "/starting")

    def mark_ready(self) -> Optional[dict]:
        return self.try_request("POST", "/ready")

    def workload_start(self, workload_id: str, label: str = "") -> Optional[dict]:
        return self.try_request(
            "POST", "/workload/start", {"id": workload_id, "label": label}
        )

    def workload_heartbeat(self, workload_id: str) -> Optional[dict]:
        return self.try_request("POST", "/workload/heartbeat", {"id": workload_id})

    def workload_end(
        self, workload_id: str, returncode: Optional[int], state: Optional[str] = None
    ) -> Optional[dict]:
        return self.try_request(
            "POST",
            "/workload/end",
            {"id": workload_id, "returncode": returncode, "state": state},
        )

    def lock(self, ttl_seconds: Optional[float], reason: str = "") -> dict:
        return self._request("POST", "/lock", {"ttl_seconds": ttl_seconds, "reason": reason})

    def unlock(self, reason: str = "manual") -> dict:
        return self._request("POST", "/unlock", {"reason": reason})

    def shutdown(self, action: str = "stop", reason: str = "manual") -> Optional[dict]:
        return self.try_request("POST", "/shutdown", {"action": action, "reason": reason})
