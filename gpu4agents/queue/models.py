"""Queue data model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Optional

from ..core.errors import JambuError


class JobState(str, Enum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    DELIVERING = "delivering"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


FINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


class Outcome(str, Enum):
    SUCCEEDED = "succeeded"
    JOB_FAILED = "job_failed"
    TIMEOUT = "timeout"
    MACHINE_LOST = "machine_lost"
    UNHEALTHY = "unhealthy"
    UNRESPONSIVE = "unresponsive"
    PREFLIGHT_REJECTED = "preflight_rejected"
    CANCELLED = "cancelled"
    DELIVERY_FAILED = "delivery_failed"
    DRAINED = "drained"  # sent back to the queue by `drain --now`: not the job's or the machine's fault
    # The attempt was recorded but its run never started (a coordinator died before the
    # dispatch): sent back to the queue, and like DRAINED it does not count toward max_attempts.
    NOT_STARTED = "not_started"


class LeaseLost(JambuError):
    """A write was made with a stale or expired claim token."""


@dataclass
class Attempt:
    number: int
    machine: Optional[str]
    remote_id: str
    started_at: float
    ended_at: Optional[float] = None
    outcome: Optional[Outcome] = None
    exit_code: Optional[int] = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value if self.outcome else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Attempt":
        d = dict(d)
        if d.get("outcome") is not None:
            d["outcome"] = Outcome(d["outcome"])
        return cls(**d)


@dataclass
class QueueJob:
    id: str
    key: str
    bundle_ref: str
    manifest: dict
    submitted_at: float
    submitted_by: str
    state: JobState = JobState.QUEUED
    attempts: list[Attempt] = field(default_factory=list)
    limits: Optional[dict] = None
    outputs_ref: Optional[dict] = None
    result: Optional[dict] = None
    cancel_requested: bool = False
    machine: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "bundle_ref": self.bundle_ref,
            "manifest": self.manifest,
            "submitted_at": self.submitted_at,
            "submitted_by": self.submitted_by,
            "state": self.state.value,
            "attempts": [a.to_dict() for a in self.attempts],
            "limits": self.limits,
            "outputs_ref": self.outputs_ref,
            "result": self.result,
            "cancel_requested": self.cancel_requested,
            "machine": self.machine,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QueueJob":
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        d["state"] = JobState(d.get("state", "queued"))
        d["attempts"] = [Attempt.from_dict(a) for a in d.get("attempts", [])]
        return cls(**d)


@dataclass
class Claim:
    job: QueueJob
    token: str
    lease_until: float


@dataclass
class QueueStats:
    by_state: dict[str, int]
    oldest_queued_age_s: Optional[float]
