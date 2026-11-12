"""The pool's scaling decision, pure: state in, decision out. It never closes a machine
(the machines' idle_timeout does that), it only says how many to open."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

COUNTED_STATES = ("provisioning", "checking", "ready", "busy")
BOOTING_STATES = ("provisioning", "checking")
LIVE_STATES = ("ready", "busy")


@dataclass
class MachineView:
    slot: str
    state: str
    hourly_usd: float
    running_jobs: int


@dataclass
class ScaleInput:
    now: float
    queued: int
    oldest_queued_age_s: Optional[float]
    machines: list[MachineView]  # oldest first
    pool_usd_per_hour: float
    next_offer_usd: Optional[float]
    min_machines: int
    max_machines: int
    max_usd_per_hour: Optional[float]
    scale_up_after_s: float
    jobs_per_machine: int
    last_open_at: Optional[float]
    open_backoff_s: float


@dataclass
class ScaleDecision:
    open: int
    keepalive: list[str] = field(default_factory=list)
    reason: str = ""
    blocked_by: Optional[str] = None  # "max_machines" | "max_usd_per_hour" | "no_offer" | "backoff" | None


def decide(inp: ScaleInput) -> ScaleDecision:
    counted = [m for m in inp.machines if m.state in COUNTED_STATES]
    count = len(counted)
    jpm = max(1, inp.jobs_per_machine)
    free = sum(max(0, jpm - m.running_jobs) for m in counted if m.state in LIVE_STATES)
    free += jpm * sum(1 for m in counted if m.state in BOOTING_STATES)

    keepalive = [m.slot for m in inp.machines if m.state == "ready" and m.running_jobs == 0][: max(0, inp.min_machines)]

    waiting = (
        inp.queued > free
        and inp.oldest_queued_age_s is not None
        and inp.oldest_queued_age_s >= inp.scale_up_after_s
    )
    for_queue = math.ceil((inp.queued - free) / jpm) if waiting else 0
    for_min = max(0, inp.min_machines - count)
    needed = max(for_queue, for_min)
    if needed == 0:
        why = "jobs waiting under the scale-up threshold" if inp.queued > free else "capacity is enough"
        return ScaleDecision(0, keepalive, why, None)

    why = f"need {needed} more ({inp.queued} queued, {free} free slots, {count} machines)"
    if inp.last_open_at is not None and inp.now - inp.last_open_at < inp.open_backoff_s:
        return ScaleDecision(0, keepalive, why, "backoff")
    if inp.next_offer_usd is None:
        return ScaleDecision(0, keepalive, why, "no_offer")

    n, blocked = needed, None
    room = inp.max_machines - count
    if room < n:
        n, blocked = max(0, room), "max_machines"
    if inp.max_usd_per_hour is not None:
        budget = inp.max_usd_per_hour - inp.pool_usd_per_hour
        affordable = int(budget / inp.next_offer_usd + 1e-9) if inp.next_offer_usd > 0 else n
        if affordable < n:
            n, blocked = max(0, affordable), "max_usd_per_hour"
    return ScaleDecision(n, keepalive, why, blocked)
