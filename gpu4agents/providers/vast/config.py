"""Vast.ai adapter settings.

Everything here is provider-local. Nothing in this module may be referenced
by the core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

API_BASE = "https://console.vast.ai/api/v0"
API_KEY_ENV = "VAST_API_KEY"

# runtype that gives us sshd plus our own onstart script (the image entrypoint
# is replaced by Vast's supervisor, so we launch the model server ourselves).
DEFAULT_RUNTYPE = "ssh_direc ssh_proxy"

# Vast rejects a create whose onstart script is longer than this.
ONSTART_MAX_BYTES = 16384

DEFAULT_SEARCH_LIMIT = 24
DEFAULT_MIN_RELIABILITY = 0.95
# Image pulls are ~10 GB; a slow pipe is the commonest boot failure. Most
# offers clear 500 Mbps at the same price.
DEFAULT_MIN_INET_DOWN_MBPS = 500.0
DEFAULT_MIN_CUDA = 12.1


@dataclass
class VastOptions:
    """`provider.options` in jambu.yaml (gpu_runtime.provider.options), all optional."""

    api_base: str = API_BASE
    runtype: str = DEFAULT_RUNTYPE
    verified_only: bool = True
    min_reliability: float = DEFAULT_MIN_RELIABILITY
    min_inet_down_mbps: float = DEFAULT_MIN_INET_DOWN_MBPS
    min_cuda: float = DEFAULT_MIN_CUDA
    search_limit: int = DEFAULT_SEARCH_LIMIT
    order_by: str = "dph_total"  # dph_total | score | inet_down
    order_dir: str = "asc"
    image_login: Optional[str] = None
    # Extra raw filters merged into the offer query, escape hatch for power users.
    search_filters: Optional[dict[str, Any]] = None

    @classmethod
    def from_mapping(cls, data: Optional[dict[str, Any]]) -> "VastOptions":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                "unknown provider.options for vast: " + ", ".join(sorted(unknown))
            )
        return cls(**{k: v for k, v in data.items() if k in known})
