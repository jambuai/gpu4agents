"""Thin HTTP client for the Lambda Cloud API (v1).

Only this module knows Lambda's wire format: every answer is wrapped in
``{"data": ...}`` and every error in ``{"error": {"code", "message"}}``. It
raises normalized errors, branching on the error code, never the message.
"""

from __future__ import annotations

from typing import Any, Optional

from ...core.errors import (
    AuthenticationError,
    CapacityUnavailableError,
    ProviderError,
    ProvisioningError,
)
from ..http import JsonClient
from .config import API_BASE


class LambdaNotFound(ProviderError):
    """The requested Lambda object does not exist (any more)."""


class LambdaClient(JsonClient):
    label = "Lambda"

    def __init__(self, api_key: str, api_base: str = API_BASE, **kwargs: Any) -> None:
        super().__init__(api_key, api_base, **kwargs)

    def request(self, method: str, path: str, *, json_body: Any = None, params: Optional[dict] = None) -> Any:
        response = self.send(method, path, json_body=json_body, params=params)
        data = self.body(response)
        if response.status_code < 400:
            return data.get("data") if isinstance(data, dict) and "data" in data else data

        error = data.get("error") if isinstance(data, dict) else None
        code = str((error or {}).get("code") or "")
        message = str((error or {}).get("message") or response.text[:300])
        if response.status_code == 401 or code == "global/invalid-api-key":
            raise AuthenticationError(
                "Lambda rejected the API key (check LAMBDA_API_KEY, or run `gpu login --provider lambda`)"
            )
        if response.status_code == 404 or code == "global/object-does-not-exist":
            raise LambdaNotFound(f"Lambda object not found: {method} {path}")
        if code == "instance-operations/launch/insufficient-capacity":
            raise CapacityUnavailableError(f"Lambda: {message}")
        if code in ("global/quota-exceeded", "global/account-inactive"):
            raise ProvisioningError(f"Lambda: {message}")
        if response.status_code == 429:
            raise ProviderError("Lambda rate limit hit; retry in a moment")
        raise ProviderError(f"Lambda error {response.status_code} ({code or 'no code'}) on {method} {path}: {message}")

    # -- catalog ------------------------------------------------------------

    def instance_types(self) -> dict[str, Any]:
        data = self.request("GET", "/instance-types")
        return data if isinstance(data, dict) else {}

    # -- instances ----------------------------------------------------------

    def launch(self, payload: dict) -> str:
        data = self.request("POST", "/instance-operations/launch", json_body=payload)
        ids = (data or {}).get("instance_ids") if isinstance(data, dict) else None
        if not ids:
            raise ProvisioningError(f"Lambda did not return an instance id: {data}")
        return str(ids[0])

    def list_instances(self) -> list[dict]:
        data = self.request("GET", "/instances")
        return list(data or []) if isinstance(data, list) else []

    def get_instance(self, instance_id: str) -> Optional[dict]:
        try:
            data = self.request("GET", f"/instances/{instance_id}")
        except LambdaNotFound:
            return None
        return data if isinstance(data, dict) else None

    def terminate(self, instance_id: str) -> None:
        try:
            self.request("POST", "/instance-operations/terminate", json_body={"instance_ids": [instance_id]})
        except LambdaNotFound:
            return

    # -- ssh keys -----------------------------------------------------------

    def ssh_keys(self) -> list[dict]:
        data = self.request("GET", "/ssh-keys")
        return list(data or []) if isinstance(data, list) else []

    def add_ssh_key(self, name: str, public_key: str) -> dict:
        data = self.request("POST", "/ssh-keys", json_body={"name": name, "public_key": public_key})
        return data if isinstance(data, dict) else {}

    # -- firewall -----------------------------------------------------------

    def firewall_rulesets(self) -> list[dict]:
        data = self.request("GET", "/firewall-rulesets")
        return list(data or []) if isinstance(data, list) else []

    def create_firewall_ruleset(self, name: str, region: str, rules: list[dict]) -> dict:
        data = self.request(
            "POST", "/firewall-rulesets", json_body={"name": name, "region": region, "rules": rules}
        )
        return data if isinstance(data, dict) else {}
