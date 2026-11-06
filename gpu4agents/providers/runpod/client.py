"""Thin HTTP client for the RunPod REST API v2.

Only this module knows RunPod's wire format. Errors arrive as RFC 9457
problem documents (``{"title", "status", "detail"}``); they are raised as
the normalized errors.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator, Optional

import httpx

from ...core.errors import (
    AuthenticationError,
    CapacityUnavailableError,
    ProviderError,
    ProvisioningError,
)
from ..http import JsonClient
from .config import API_BASE, GRAPHQL_URL


class RunpodNotFound(ProviderError):
    """The requested pod does not exist (any more)."""


class RunpodConflict(ProviderError):
    """The action does not fit the pod's current status (HTTP 409)."""


class RunpodClient(JsonClient):
    label = "RunPod"

    def __init__(
        self, api_key: str, api_base: str = API_BASE, graphql_url: str = GRAPHQL_URL, **kwargs: Any
    ) -> None:
        super().__init__(api_key, api_base, **kwargs)
        self.graphql_url = graphql_url

    def request(
        self, method: str, path: str, *, json_body: Any = None, params: Optional[dict] = None, creating: bool = False
    ) -> Any:
        response = self.send(method, path, json_body=json_body, params=params)
        data = self.body(response)
        status = response.status_code
        if status < 400:
            return data if data is not None else {}
        detail = ""
        if isinstance(data, dict):
            detail = str(data.get("detail") or data.get("title") or data.get("error") or "")
        detail = detail or response.text[:300]
        if status == 401:
            raise AuthenticationError(
                "RunPod rejected the API key (check RUNPOD_API_KEY, or run `gpu login --provider runpod`)"
            )
        if status == 403:
            message = f"RunPod refused: {detail}"
            if creating and not any(w in detail.lower() for w in ("scope", "permission", "api key", "token")):
                # The account may not use that cloud or GPU pool: try the next one.
                raise CapacityUnavailableError(message)
            raise AuthenticationError(message + " (the key may lack permission: create one with All access)")
        if status == 404:
            raise RunpodNotFound(f"RunPod object not found: {method} {path}")
        if status == 409:
            raise RunpodConflict(f"RunPod: {detail}")
        if status == 402:
            raise ProvisioningError(f"RunPod: not enough credit ({detail}). Add credit and retry.")
        if status == 400 and creating:
            # RunPod answers "no capacity" and a rule violation the same way.
            raise CapacityUnavailableError(f"RunPod: {detail}")
        if status == 429:
            raise ProviderError("RunPod rate limit hit; retry in a moment")
        raise ProviderError(f"RunPod error {status} on {method} {path}: {detail}")

    # -- account ------------------------------------------------------------

    def balance(self) -> Optional[float]:
        """The account's credit, from GraphQL (the only API that has it)."""
        response = self.send(
            "POST",
            self.graphql_url,
            json_body={"query": "query { myself { clientBalance } }"},
            params={"api_key": self.api_key},
        )
        if response.status_code in (401, 403):
            raise AuthenticationError(
                "RunPod rejected the API key (check RUNPOD_API_KEY, or run `gpu login --provider runpod`)"
            )
        data = self.body(response)
        if not isinstance(data, dict) or response.status_code >= 400:
            return None
        errors = data.get("errors") or []
        if errors and any("unauthorized" in str(e).lower() or "api key" in str(e).lower() for e in errors):
            raise AuthenticationError("RunPod rejected the API key")
        myself = (data.get("data") or {}).get("myself") or {}
        try:
            return round(float(myself.get("clientBalance")), 2)
        except (TypeError, ValueError):
            return None

    # -- catalog ------------------------------------------------------------

    def gpu_types(self, count: int, cloud: str, min_cuda: Optional[str] = None) -> list[dict]:
        params: dict[str, Any] = {"include": "AVAILABILITY", "product": "POD", "count": int(count), "cloud": cloud}
        if min_cuda:
            params["minCudaVersion"] = min_cuda
        data = self.request("GET", "/v2/catalog/gpus", params=params)
        return list(data.get("gpus") or []) if isinstance(data, dict) else []

    # -- pods -----------------------------------------------------------------

    def create_pod(self, payload: dict) -> dict:
        data = self.request("POST", "/v2/pods", json_body=payload, creating=True)
        if not isinstance(data, dict) or not data.get("id"):
            raise ProvisioningError(f"RunPod did not return a pod id: {data}")
        return data

    def get_pod(self, pod_id: str) -> Optional[dict]:
        try:
            data = self.request("GET", f"/v2/pods/{pod_id}")
        except RunpodNotFound:
            return None
        return data if isinstance(data, dict) else None

    def list_pods(self) -> list[dict]:
        pods: list[dict] = []
        cursor: Optional[str] = None
        for _ in range(20):
            params: dict[str, Any] = {"limit": 1000}
            if cursor:
                params["cursor"] = cursor
            data = self.request("GET", "/v2/pods", params=params)
            if not isinstance(data, dict):
                break
            pods.extend(data.get("pods") or [])
            page = data.get("pagination") or {}
            cursor = page.get("nextCursor")
            if not page.get("hasNextPage") or not cursor:
                break
        return pods

    def action(self, pod_id: str, action: str) -> dict:
        data = self.request("POST", f"/v2/pods/{pod_id}/action", json_body={"action": action})
        return data if isinstance(data, dict) else {}

    def delete_pod(self, pod_id: str) -> None:
        try:
            self.request("DELETE", f"/v2/pods/{pod_id}")
        except RunpodNotFound:
            return

    def logs(self, pod_id: str, tail: int = 800, wait_s: float = 8.0) -> Iterator[str]:
        """The container log's last lines. The route streams (SSE); read for ``wait_s`` at most."""
        url = f"{self.api_base}/v2/pods/{pod_id}/logs"
        deadline = time.time() + wait_s
        try:
            with self.client.stream(
                "GET",
                url,
                params={"source": "container", "tail": int(tail)},
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(wait_s, read=3.0),
            ) as response:
                if response.status_code >= 400:
                    raise ProviderError(f"RunPod log request failed ({response.status_code})")
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        try:
                            event = json.loads(raw)
                        except ValueError:
                            event = {"line": raw}
                        if isinstance(event, dict):
                            yield str(event.get("line") or "")
                    if time.time() > deadline:
                        return
        except httpx.ReadTimeout:
            return  # the backlog was sent; the stream would only wait for new lines
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not read RunPod logs: {exc}") from exc
