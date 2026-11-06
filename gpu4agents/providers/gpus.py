"""What adapters know about GPU models that some provider APIs do not say.

Vast.ai reports each offer's compute capability; RunPod and Lambda report
only a model name. The runtime needs the architecture (an image built for
Ampere dies on a Volta card), so adapters look the name up here. A name not
listed returns None, and the offer is not filtered on it.
"""

from __future__ import annotations

import re
from typing import Optional

# Substring of the GPU name (lowercase, spaces removed) -> compute capability.
# Longest keys are tried first, so "a100" never shadows "a1000".
_COMPUTE_CAP = {
    "b300": 10.3,
    "b200": 10.0,
    "gb200": 10.0,
    "rtx5090": 12.0,
    "rtx5080": 12.0,
    "rtxpro6000": 12.0,
    "h200": 9.0,
    "h100": 9.0,
    "gh200": 9.0,
    "l40s": 8.9,
    "l40": 8.9,
    "l4": 8.9,
    "rtx4090": 8.9,
    "rtx4080": 8.9,
    "rtx4070": 8.9,
    "rtx6000ada": 8.9,
    "rtx5000ada": 8.9,
    "rtx4000ada": 8.9,
    "rtx2000ada": 8.9,
    "a100": 8.0,
    "a30": 8.0,
    "a40": 8.6,
    "a10g": 8.6,
    "a10": 8.6,
    "a16": 8.6,
    "a6000": 8.6,
    "a5000": 8.6,
    "a4500": 8.6,
    "a4000": 8.6,
    "a2000": 8.6,
    "rtx3090": 8.6,
    "rtx3080": 8.6,
    "rtx3070": 8.6,
    "rtx3060": 8.6,
    "quadrortx6000": 7.5,
    "rtx6000": 7.5,  # Quadro RTX 6000 (Turing); the Ada and Pro cards match longer keys
    "quadrortx8000": 7.5,
    "rtx2080": 7.5,
    "t4": 7.5,
    "v100": 7.0,
    "p100": 6.0,
    "p40": 6.1,
    "p4": 6.1,
}
_ORDERED = sorted(_COMPUTE_CAP, key=len, reverse=True)


def compute_cap(gpu_name: str) -> Optional[float]:
    """The compute capability of a GPU model named like "NVIDIA H100 80GB HBM3"."""
    key = re.sub(r"[\s_\-]+", "", (gpu_name or "").lower()).replace("nvidia", "").replace("geforce", "")
    for needle in _ORDERED:
        if needle in key:
            return _COMPUTE_CAP[needle]
    return None
