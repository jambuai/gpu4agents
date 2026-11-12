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


def _number(manifest: dict, key: str) -> Optional[float]:
    """A finite number from the manifest, None when absent; ValueError when malformed."""
    value = manifest.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return float(value)


def default_preflight(job: QueueJob, ctx: PreflightContext) -> PreflightResult:
    manifest = job.manifest or {}
    try:
        raw_timeout = manifest.get("timeout")
        if isinstance(raw_timeout, bool):
            raise ValueError(f"timeout must be a duration, got {raw_timeout!r}")
        timeout_s = parse_duration(raw_timeout)
        if timeout_s is None:
            timeout_s = parse_duration(ctx.defaults.timeout)
        margin_s = parse_duration(ctx.defaults.deadline_margin) or 0.0
        max_cost = _number(manifest, "max_cost_usd")
        need = _number(manifest, "min_vram_gb")
    except (TypeError, ValueError, AttributeError) as exc:
        return _reject(f"malformed job value: {exc}")
    if timeout_s is None:
        return _reject("no timeout in the job or the queue defaults")
    if timeout_s <= 0:
        return _reject(f"timeout must be positive, got {timeout_s:g}s")
    if max_cost is None:
        max_cost = ctx.defaults.max_cost_usd
    if max_cost is not None and max_cost < 0:
        return _reject(f"max_cost_usd must not be negative, got {max_cost:g}")
    if need is not None and need < 0:
        return _reject(f"min_vram_gb must not be negative, got {need:g}")

    if need is not None and ctx.card_vram_gb is not None and need > ctx.card_vram_gb:
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
    except Exception as exc:  # import-time errors in the project's module too
        raise ConfigError(f"cannot load preflight {spec!r}: {exc}") from exc
    if not callable(fn):
        raise ConfigError(f"preflight {spec!r} is not callable")

    def safe(job: QueueJob, ctx: PreflightContext) -> PreflightResult:
        try:
            result = fn(job, ctx)
        except Exception as exc:  # the coordinator must never crash on a project's rule
            return _reject(f"preflight {spec} failed: {exc}")
        if not isinstance(result, PreflightResult):
            return _reject(f"preflight {spec} returned {type(result).__name__}, not PreflightResult")
        if result.ok and not isinstance(result.limits, Limits):
            return _reject(f"preflight {spec} accepted the job without Limits")
        if not result.ok and not result.reason:
            return _reject(f"preflight {spec} rejected the job without a reason")
        return result

    return safe
