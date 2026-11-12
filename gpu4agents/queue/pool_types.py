"""Types shared by the pool, the scaler's input and spend accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class MachineState:
    PROVISIONING = "provisioning"
    CHECKING = "checking"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    UNHEALTHY = "unhealthy"
    UNRESPONSIVE = "unresponsive"
    GONE = "gone"
    RETIRED = "retired"
    STOPPED = "stopped"
    HELD = "held"  # kept running so outputs can still be pulled

    ALL = (
        PROVISIONING, CHECKING, READY, BUSY, DRAINING, UNHEALTHY,
        UNRESPONSIVE, GONE, RETIRED, STOPPED, HELD,
    )


@dataclass
class PoolMachine:
    slot: str
    instance_id: Optional[str]
    state: str
    hourly_usd: float
    stopped_hourly_usd: float
    ready_at: Optional[float]
    deadline_at: Optional[float]
    jobs: list[str] = field(default_factory=list)
    health_failures: int = 0
    last_seen_at: Optional[float] = None
    opened_at: Optional[float] = None
