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


# Rough speed of each GPU model on deep-learning work (bf16/fp16 tensor
# throughput, sanity-checked against diffusion and LLM benchmarks), with an
# RTX 4090 as 1.0. Good enough to rank offers by what a whole job costs, not
# to promise a time: `gpu offers --for` labels its figures as estimates.
_RELATIVE_SPEED = {
    "b300": 3.2,
    "b200": 3.0,
    "gb200": 3.0,
    "h200": 2.4,
    "gh200": 2.4,
    "h100": 2.1,
    "rtxpro6000": 1.4,
    "rtx5090": 1.4,
    "rtx5080": 0.9,
    "l40s": 1.0,
    "l40": 0.9,
    "l4": 0.3,
    "a100": 1.0,
    "a1000": 0.15,
    "rtx6000ada": 1.0,
    "rtx5000ada": 0.7,
    "rtx4000ada": 0.35,
    "rtx2000ada": 0.2,
    "rtx4090": 1.0,
    "rtx4080": 0.75,
    "rtx4070ti": 0.55,
    "rtx4070": 0.45,
    "rtx4060ti": 0.3,
    "rtx4060": 0.22,
    "a6000": 0.6,
    "a5000": 0.45,
    "a4500": 0.4,
    "a4000": 0.3,
    "a40": 0.55,
    "a10g": 0.35,
    "a10": 0.4,
    "a16": 0.1,
    "rtx3090": 0.55,
    "rtx3080": 0.45,
    "rtx3070": 0.3,
    "rtx3060": 0.25,
    "quadrortx6000": 0.35,
    "rtx6000": 0.35,
    "quadrortx8000": 0.35,
    "rtx2080": 0.25,
    "rtx2070": 0.18,
    "rtx2060": 0.15,
    "t4": 0.12,
    "v100": 0.35,
    "p100": 0.12,
    "p40": 0.08,
}
_SPEED_ORDER = sorted(_RELATIVE_SPEED, key=len, reverse=True)


def relative_speed(gpu_name: str) -> Optional[float]:
    """How fast a GPU model is next to an RTX 4090 (1.0), or None if unknown."""
    key = re.sub(r"[\s_\-]+", "", (gpu_name or "").lower()).replace("nvidia", "").replace("geforce", "")
    for needle in _SPEED_ORDER:
        if needle in key:
            return _RELATIVE_SPEED[needle]
    return None
