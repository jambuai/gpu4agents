"""Lambda Cloud adapter settings.

Everything here is provider-local. Nothing in this module may be referenced
by the core. (The package is not called ``lambda``: that is a Python keyword.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

API_BASE = "https://cloud.lambda.com/api/v1"
API_KEY_ENV = "LAMBDA_API_KEY"
KEYS_URL = "https://cloud.lambda.com/api-keys"
BILLING_URL = "https://cloud.lambda.com/settings/billing"
SIGNUP_URL = "https://cloud.lambda.com/sign-up"

# Lambda launches one machine per call and limits launches to one every 12 s.
MAX_LAUNCH_ATTEMPTS = 4

# The key gpu registers with Lambda for SSH (one key per launch is required).
SSH_KEY_NAME_PREFIX = "gpu4agents-"
# Name prefix of the per-region firewall rulesets gpu creates for the ports
# it needs (model, watchdog) on top of SSH.
RULESET_NAME_PREFIX = "gpu4agents-"
# Tag key that carries a per-launch nonce: the watchdog finds its own machine
# by it, since a Lambda VM cannot read its own instance id.
NONCE_TAG = "gpu4agents-nonce"

# Where gpu keeps the SSH key it uses for Lambda machines.
SSH_KEY_FILE = "ssh/lambda_ed25519"


@dataclass
class LambdaOptions:
    """`provider.options` in gpu.yaml for lambda, all optional."""

    api_base: str = API_BASE
    # Image family, e.g. lambda-stack-22-04. Unset: Lambda's default for the
    # instance type, which has Docker and the NVIDIA container toolkit.
    image_family: Optional[str] = None
    # Arm64 machines (GH200) cannot run the usual x86 model images.
    allow_arm64: bool = False
    # Who may reach the model and watchdog ports. The model demands an API
    # key and the watchdog a token, as on every provider.
    source_network: str = "0.0.0.0/0"

    @classmethod
    def from_mapping(cls, data: Optional[dict[str, Any]]) -> "LambdaOptions":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError("unknown provider.options for lambda: " + ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in data.items() if k in known})
