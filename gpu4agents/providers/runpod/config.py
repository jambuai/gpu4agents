"""RunPod adapter settings (REST API v2).

Everything here is provider-local. Nothing in this module may be referenced
by the core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

API_BASE = "https://api.runpod.io"
# Only the account balance still needs GraphQL; v2 has no balance route.
GRAPHQL_URL = "https://api.runpod.io/graphql"
API_KEY_ENV = "RUNPOD_API_KEY"
KEYS_URL = "https://console.runpod.io/user/credentials"
BILLING_URL = "https://console.runpod.io/user/billing"
# Where `gpu login` sends someone with no RunPod account. Set REFERRAL_LINK
# to the link under the console's Referrals page (https://runpod.io?ref=...):
# RunPod then credits the project with a share of what it earns from that
# account, at no cost to the user. Empty: the plain sign-up page.
REFERRAL_LINK = ""
SIGNUP_URL = REFERRAL_LINK or "https://console.runpod.io/signup"

# Pod create bodies are capped at 100 KiB.
MAX_BODY_BYTES = 102_400
# Offers tried, in price order, before giving up on a setup.
MAX_CREATE_ATTEMPTS = 6

CLOUDS = ("SECURE", "COMMUNITY")


@dataclass
class RunpodOptions:
    """`provider.options` in gpu.yaml for runpod, all optional."""

    api_base: str = API_BASE
    graphql_url: str = GRAPHQL_URL
    # any | secure (RunPod's own data centers) | community (hosts it vets).
    cloud: str = "any"
    # GB of persistent volume at /workspace. 0: none, so a stopped pod costs
    # nothing, but the model downloads again when it starts.
    volume_gb: int = 0
    # Skip GPU types RunPod reports as scarce (LOW) for the requested count.
    skip_low_availability: bool = False

    @classmethod
    def from_mapping(cls, data: Optional[dict[str, Any]]) -> "RunpodOptions":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError("unknown provider.options for runpod: " + ", ".join(sorted(unknown)))
        options = cls(**{k: v for k, v in data.items() if k in known})
        if options.cloud.lower() not in ("any", "secure", "community"):
            raise ValueError("provider.options.cloud for runpod must be any, secure or community")
        return options

    def clouds(self) -> tuple[str, ...]:
        wanted = self.cloud.lower()
        return CLOUDS if wanted == "any" else (wanted.upper(),)
