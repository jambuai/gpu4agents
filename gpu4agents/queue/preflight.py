"""Per-job preflight: effective limits and refusals that explain themselves."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Optional

from ..core.config import QueueDefaults
from ..core.durations import parse_duration
from ..core.errors import ConfigError
from .models import QueueJob


@dataclass
class Limits:
    timeout_s: float
    max_cost_usd: Optional[float]
    deadline_margin_s: float


@dataclass
class PreflightContext:
    hourly_usd: float
    card_vram_gb: Optional[float]
    defaults: QueueDefaults
    attempt: int


@dataclass
class PreflightResult:
    ok: bool
    limits: Optional[Limits]
    reason: str = ""


Preflight = Callable[[QueueJob, PreflightContext], PreflightResult]


def _reject(reason: str) -> PreflightResult:
    return PreflightResult(ok=False, limits=None, reason=reason)


def default_preflight(job: QueueJob, ctx: PreflightContext) -> PreflightResult:
    manifest = job.manifest or {}
    try:
        timeout_s = parse_duration(manifest.get("timeout"))
        if timeout_s is None:
            timeout_s = parse_duration(ctx.defaults.timeout)
        margin_s = parse_duration(ctx.defaults.deadline_margin) or 0.0
    except ValueError as exc:
        return _reject(str(exc))
    if timeout_s is None:
        return _reject("no timeout in the job or the queue defaults")

    max_cost = manifest.get("max_cost_usd")
    if max_cost is None:
        max_cost = ctx.defaults.max_cost_usd
    max_cost = float(max_cost) if max_cost is not None else None

    need = manifest.get("min_vram_gb")
    if need is not None and ctx.card_vram_gb is not None and float(need) > ctx.card_vram_gb:
        return _reject(f"needs {need:g} GB, the pool's card has {ctx.card_vram_gb:g} GB")

    if max_cost is not None:
        worst = ctx.hourly_usd * timeout_s / 3600
        if max_cost < worst:
            return _reject(
                f"max_cost_usd {max_cost:g} is below {worst:g} "
                f"(${ctx.hourly_usd:g}/h for the {timeout_s:g}s timeout)"
            )
    return PreflightResult(True, Limits(timeout_s, max_cost, margin_s))


def load_preflight(spec: Optional[str]) -> Preflight:
    """None -> the default; "module:function" -> that function, wrapped so it never raises."""
    if not spec:
        return default_preflight
    if ":" not in spec:
        raise ConfigError(f"unknown preflight {spec!r}; use module:function")
    module_name, _, attr = spec.partition(":")
    try:
        fn = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot load preflight {spec!r}: {exc}") from exc

    def safe(job: QueueJob, ctx: PreflightContext) -> PreflightResult:
        try:
            result = fn(job, ctx)
        except Exception as exc:  # the coordinator must never crash on a project's rule
            return _reject(f"preflight {spec} failed: {exc}")
        if not isinstance(result, PreflightResult):
            return _reject(f"preflight {spec} returned {type(result).__name__}, not PreflightResult")
        return result

    return safe
