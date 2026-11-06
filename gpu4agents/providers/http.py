"""A small JSON-over-HTTP client the RunPod and Lambda adapters share.

It retries what is safe to retry (a rate limit on any call, a gateway error
on a read) and turns transport failures into the normalized errors. Each
adapter's client classifies the provider's own error bodies.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from .. import __version__
from ..core.errors import ProviderError, ProviderTimeoutError

# Waits between attempts on a rate limit or a transient gateway error.
RETRY_DELAYS_S = (2.0, 4.0, 8.0, 15.0, 15.0)


def retry_after(response: httpx.Response, default: float) -> float:
    try:
        return min(60.0, max(0.0, float(response.headers.get("Retry-After", default))))
    except ValueError:
        return default


class JsonClient:
    """Bearer-authenticated JSON client. Subclasses set ``label``."""

    label = "provider"

    def __init__(
        self,
        api_key: str,
        api_base: str,
        timeout: float = 60.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self._client = client
        self.sleep = time.sleep

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

    def send(
        self,
        method: str,
        url: str,
        *,
        json_body: Any = None,
        params: Optional[dict] = None,
    ) -> httpx.Response:
        """The response after retries; status codes are left to the caller."""
        if not url.startswith("http"):
            url = f"{self.api_base}{url}"
        response: Optional[httpx.Response] = None
        for delay in (*RETRY_DELAYS_S, None):
            try:
                response = self.client.request(method, url, json=json_body, params=params)
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(f"{self.label} timed out on {method} {url}") from exc
            except httpx.HTTPError as exc:
                raise ProviderError(f"{self.label} request failed ({method} {url}): {exc}") from exc
            # A 429 was not processed, so any call can be resent; a 5xx only
            # a read, since a create may have gone through.
            transient = response.status_code == 429 or (
                method == "GET" and response.status_code in (502, 503, 504)
            )
            if not transient or delay is None:
                break
            self.sleep(retry_after(response, delay))
        assert response is not None
        return response

    @staticmethod
    def body(response: httpx.Response) -> Any:
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return None
