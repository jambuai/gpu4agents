"""Thin HTTP client for the Vast.ai REST API.

Only this module knows Vast's wire format. It raises normalized errors so
nothing Vast-shaped escapes the adapter.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx

from ... import __version__
from ...core.errors import (
    AuthenticationError,
    CapacityUnavailableError,
    ProviderError,
    ProviderTimeoutError,
    ProvisioningError,
)
from .config import API_BASE


class VastNotFound(ProviderError):
    """The requested Vast object does not exist (any more)."""


# Waits between attempts on a rate limit or a transient gateway error: about
# 45 seconds in all before giving up.
RETRY_DELAYS_S = (2.0, 4.0, 8.0, 15.0, 15.0)


def _retry_after(response: httpx.Response, default: float) -> float:
    try:
        return min(60.0, max(0.0, float(response.headers.get("Retry-After", default))))
    except ValueError:
        return default


class VastClient:
    def __init__(
        self,
        api_key: str,
        api_base: str = API_BASE,
        timeout: float = 60.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self._client = client
        self.sleep = time.sleep

    # -- transport ----------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": f"gpu4agents/{__version__}",
                },
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Any:
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        for delay in (*RETRY_DELAYS_S, None):
            try:
                response = self.client.request(method, url, json=json_body, params=params)
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(f"vast.ai timed out on {method} {path}") from exc
            except httpx.HTTPError as exc:
                raise ProviderError(f"vast.ai request failed ({method} {path}): {exc}") from exc
            # Vast rate-limits bursts (a sweep timer and a setup at once are
            # enough). A 429 was not processed, so any method can be resent;
            # a 5xx only for reads, since a create may have gone through.
            transient = response.status_code == 429 or (
                method == "GET" and response.status_code in (502, 503, 504)
            )
            if not transient or delay is None:
                break
            self.sleep(_retry_after(response, delay))

        # These two are reliable regardless of what's in the body.
        if response.status_code == 404:
            raise VastNotFound(f"vast.ai object not found: {method} {path}")
        if response.status_code in (401, 403):
            raise AuthenticationError(
                "vast.ai rejected the API key (check VAST_API_KEY at "
                "https://cloud.vast.ai/account/)"
            )

        # Vast returns its own {"success": false, "msg": "..."} error shape
        # on other 4xx responses just as often as on 200 - the old behavior
        # (raise a blunt ProviderError for any >= 400 BEFORE ever looking at
        # the body) intercepted real, specifically classifiable errors like
        # "no_such_ask ... is not available" and turned them into a generic
        # ProviderError that setup()'s offer-retry loop can't walk past.
        # Parse the body first and classify by its content when there is one;
        # only fall back to the bare status code when there isn't.
        data: Any = None
        if response.content:
            try:
                data = response.json()
            except (ValueError, json.JSONDecodeError):
                data = None

        if isinstance(data, dict) and data.get("success") is False:
            message = str(data.get("msg") or data.get("error") or data)
            # Vast spells the same "this offer is gone" failure differently
            # across endpoints/versions ("no_such_ask" underscored, "no such
            # ask" spaced, "is not available" two words, "unavailable" one) -
            # normalize separators before matching so none of them slip
            # through as a generic ProviderError.
            normalized = message.lower().replace("_", " ").replace("-", " ")
            lowered = message.lower()
            if (
                "no such ask" in normalized
                or "not available" in normalized
                or "unavailable" in normalized
                or "taken" in normalized
                or "no longer" in normalized
            ):
                raise CapacityUnavailableError(f"vast.ai: {message}")
            if "insufficient" in lowered or "credit" in lowered or "balance" in lowered:
                raise ProvisioningError(f"vast.ai: {message}")
            raise ProviderError(f"vast.ai: {message}")

        if response.status_code == 429:
            raise ProviderError("vast.ai rate limit hit; retry in a moment")
        if response.status_code >= 400:
            raise ProviderError(
                f"vast.ai error {response.status_code} on {method} {path}: "
                f"{response.text[:400]}"
            )

        if data is None:
            return {"raw": response.text} if response.content else {}
        return data

    # -- account ------------------------------------------------------------

    def whoami(self) -> dict:
        data = self.request("GET", "/users/current/")
        return data if isinstance(data, dict) else {}

    # -- offers -------------------------------------------------------------

    def search_offers(self, query: dict) -> list[dict]:
        """Vast has shipped three shapes of this endpoint; try them in order.

        Stops at the first shape that *answers in the expected shape*
        (a dict with an "offers" key), even when that list is legitimately
        empty - a correct "no matches" MUST NOT fall through to the next
        (possibly incompatible) shape, or a real empty result gets replaced
        by a confusing error from a fallback shape that was never going to
        work. Only a shape that outright fails (raises) is skipped.
        """
        attempts = (
            ("PUT", "/search/asks/", {"q": query}, None),
            ("PUT", "/search/asks/", query, None),
            ("GET", "/bundles/", None, {"q": json.dumps(query)}),
        )
        last_error: Optional[Exception] = None
        for method, path, body, params in attempts:
            try:
                data = self.request(method, path, json_body=body, params=params)
            except (ProviderError, VastNotFound) as exc:
                last_error = exc
                continue
            if isinstance(data, dict) and "offers" in data:
                return list(data["offers"] or [])
        if last_error is not None and not isinstance(last_error, VastNotFound):
            raise last_error
        return []

    # -- instances ----------------------------------------------------------

    def create_instance(self, ask_id: str, payload: dict) -> dict:
        data = self.request("PUT", f"/asks/{ask_id}/", json_body=payload)
        if not isinstance(data, dict) or not data.get("new_contract"):
            raise ProvisioningError(f"vast.ai did not return an instance id: {data}")
        return data

    def list_instances(self) -> list[dict]:
        data = self.request("GET", "/instances/", params={"owner": "me"})
        if isinstance(data, dict):
            return list(data.get("instances") or [])
        return []

    def get_instance(self, instance_id: str) -> Optional[dict]:
        try:
            data = self.request("GET", f"/instances/{instance_id}/")
        except VastNotFound:
            return None
        if isinstance(data, dict):
            found = data.get("instances")
            if isinstance(found, list):
                return found[0] if found else None
            if isinstance(found, dict):
                return found or None
        # Some deployments answer the single-instance route with a bare object.
        for candidate in self.list_instances():
            if str(candidate.get("id")) == str(instance_id):
                return candidate
        return None

    def set_instance_state(self, instance_id: str, state: str) -> dict:
        data = self.request("PUT", f"/instances/{instance_id}/", json_body={"state": state})
        return data if isinstance(data, dict) else {}

    def destroy_instance(self, instance_id: str) -> dict:
        try:
            data = self.request("DELETE", f"/instances/{instance_id}/")
        except VastNotFound:
            return {"success": True, "already_gone": True}
        return data if isinstance(data, dict) else {}

    def request_logs(self, instance_id: str, tail: int = 500) -> Optional[str]:
        data = self.request(
            "PUT",
            f"/instances/request_logs/{instance_id}/",
            json_body={"tail": str(tail)},
        )
        if not isinstance(data, dict):
            return None
        return data.get("result_url")

    def fetch_url(self, url: str) -> str:
        try:
            response = httpx.get(url, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not download vast.ai logs: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"vast.ai log download failed ({response.status_code})")
        return response.text
