"""Lifecycle policy (spec sections 10-13, 25).

This module is the SINGLE SOURCE OF TRUTH for "may this instance stay
billable?".  It is deliberately stdlib-only and dependency-free because its
source is inlined verbatim into the remote watchdog agent that runs on the
rented instance (see gpu4agents/watchdog/builder.py).  Do not import anything
from the rest of the package here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# --- normalized runtime state (spec section 8) -------------------------------

PROVISIONING = "provisioning"
STARTING = "starting"
READY = "ready"
BUSY = "busy"
IDLE = "idle"
STOPPED = "stopped"
FAILED = "failed"
DESTROYED = "destroyed"
UNKNOWN = "unknown"

# --- actions -----------------------------------------------------------------

ACTION_NONE = "none"
ACTION_STOP = "stop"
ACTION_DESTROY = "destroy"

# --- reasons (recorded on every automatic shutdown, spec criterion 12) --------

REASON_IDLE = "idle_timeout"
REASON_MAX_LIFETIME = "max_lifetime"
REASON_BUDGET = "budget_exceeded"
REASON_STOPPED_TTL = "stopped_ttl"
REASON_DEADLINE = "deadline"


@dataclass
class Workload:
    """A registered unit of useful work (spec section 12)."""

    id: str
    started_at: float
    last_heartbeat: float
    label: str = ""
    state: str = "running"  # running | queued | finished | failed

    def is_pending(self) -> bool:
        return self.state in ("running", "queued")

    def is_stale(self, now: float, grace: float) -> bool:
        """Heartbeat vanished for longer than the grace period (spec section 13)."""
        if not self.is_pending():
            return False
        if self.state == "queued":
            return False
        return (now - self.last_heartbeat) > grace


@dataclass
class PolicyConfig:
    idle_timeout_s: Optional[float] = None
    max_lifetime_s: Optional[float] = None
    heartbeat_grace_s: float = 300.0
    auto_stop: bool = True
    auto_destroy: bool = False
    allow_indefinite_lock: bool = False
    stop_supported: bool = True
    destroy_supported: bool = True
    # False when a stopped instance keeps costing real money (spec section 25).
    stop_eliminates_billing: bool = True
    # Cost guardrails (spec section 17).
    hourly_cost_usd: Optional[float] = None
    max_session_cost_usd: Optional[float] = None
    # A stopped instance still bills storage; destroy it after this long.
    stopped_ttl_s: Optional[float] = None
    # Stop (keep the disk, fast restart) even where a stopped instance still
    # bills. Off by default: nothing on a stopped machine can remove it later,
    # so stopped_ttl would depend on the user's computer running `gpu sweep`.
    keep_disk: bool = False
    deadline_enforced: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]


@dataclass
class Snapshot:
    """Everything the policy needs to decide. Times are unix seconds."""

    now: float
    created_at: float
    last_activity: float
    workloads: list = field(default_factory=list)  # list[Workload]
    starting: bool = False
    locked: bool = False
    locked_until: Optional[float] = None
    provider_state: str = READY
    # Start of the current container session. A restarted instance gets a new
    # one, so max_lifetime and the budget do not kill it the moment it boots.
    session_started_at: Optional[float] = None
    # When the provider last reported the instance stopped (CLI side only).
    stopped_since: Optional[float] = None
    deadline_at: Optional[float] = None

    # -- derived helpers ----------------------------------------------------

    def lock_active(self, cfg: PolicyConfig) -> bool:
        if not self.locked:
            return False
        if self.locked_until is None:
            # An indefinite lock only holds when the project explicitly allows it.
            return bool(cfg.allow_indefinite_lock)
        return self.locked_until > self.now

    def lock_expired(self, cfg: PolicyConfig) -> bool:
        if not self.locked:
            return False
        if self.locked_until is None:
            return not cfg.allow_indefinite_lock
        return self.locked_until <= self.now

    def pending(self, cfg: PolicyConfig) -> list:
        """Workloads that still count as useful work."""
        return [
            w
            for w in self.workloads
            if w.is_pending() and not w.is_stale(self.now, cfg.heartbeat_grace_s)
        ]

    def stale(self, cfg: PolicyConfig) -> list:
        return [w for w in self.workloads if w.is_stale(self.now, cfg.heartbeat_grace_s)]

    def idle_seconds(self, cfg: PolicyConfig) -> float:
        """Seconds since the last thing that counted as activity."""
        if self.pending(cfg) or self.starting:
            return 0.0
        marks = [self.last_activity]
        for w in self.workloads:
            marks.append(w.last_heartbeat)
        return max(0.0, self.now - max(marks))

    def age_seconds(self) -> float:
        return max(0.0, self.now - self.created_at)

    def session_age_seconds(self) -> float:
        start = self.session_started_at if self.session_started_at is not None else self.created_at
        return max(0.0, self.now - start)

    def stopped_seconds(self) -> Optional[float]:
        if self.stopped_since is None:
            return None
        return max(0.0, self.now - self.stopped_since)


@dataclass
class Decision:
    action: str = ACTION_NONE
    reason: Optional[str] = None
    detail: dict = field(default_factory=dict)

    @property
    def terminates(self) -> bool:
        return self.action in (ACTION_STOP, ACTION_DESTROY)

    def to_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason, "detail": dict(self.detail)}


def derive_state(snapshot: Snapshot, cfg: PolicyConfig) -> str:
    """Normalized observed state of a live instance."""
    if snapshot.provider_state in (STOPPED, DESTROYED, FAILED, PROVISIONING):
        return snapshot.provider_state
    if snapshot.starting:
        return STARTING
    if snapshot.pending(cfg):
        return BUSY
    if snapshot.provider_state == READY:
        return IDLE if snapshot.idle_seconds(cfg) > 0 else READY
    return snapshot.provider_state


def _terminal_action(cfg: PolicyConfig) -> str:
    """Pick the cheapest supported action that actually stops the bleeding."""
    if cfg.auto_destroy:
        return ACTION_DESTROY if cfg.destroy_supported else ACTION_NONE
    if not cfg.auto_stop:
        return ACTION_NONE
    if cfg.stop_supported and (cfg.stop_eliminates_billing or cfg.keep_disk):
        return ACTION_STOP
    # A stop would leave a bill behind (or is impossible): destroy.
    if cfg.destroy_supported:
        return ACTION_DESTROY
    return ACTION_STOP if cfg.stop_supported else ACTION_NONE


def evaluate(snapshot: Snapshot, cfg: PolicyConfig) -> Decision:
    """Core lifecycle rule (spec section 10). Pure function, no side effects."""
    detail: dict[str, Any] = {
        "state": derive_state(snapshot, cfg),
        "age_seconds": round(snapshot.age_seconds(), 1),
        "session_age_seconds": round(snapshot.session_age_seconds(), 1),
        "idle_seconds": round(snapshot.idle_seconds(cfg), 1),
        "pending_workloads": len(snapshot.pending(cfg)),
        "stale_workloads": len(snapshot.stale(cfg)),
        "locked": snapshot.lock_active(cfg),
    }

    locked = snapshot.lock_active(cfg)

    if snapshot.provider_state == STOPPED:
        stopped_for = snapshot.stopped_seconds()
        if (
            cfg.stopped_ttl_s is not None
            and stopped_for is not None
            and stopped_for > cfg.stopped_ttl_s
            and cfg.destroy_supported
            and not locked
        ):
            detail["stopped_seconds"] = round(stopped_for, 1)
            return Decision(ACTION_DESTROY, REASON_STOPPED_TTL, detail)
        return Decision(ACTION_NONE, "already_terminal", detail)
    if snapshot.provider_state == DESTROYED:
        return Decision(ACTION_NONE, "already_terminal", detail)

    # Queue-mode deadline: ends the machine under work and over a lock.
    d = snapshot.deadline_at
    if cfg.deadline_enforced and d is not None and snapshot.now > d:
        action = _terminal_action(cfg)
        if action != ACTION_NONE:
            return Decision(action, REASON_DEADLINE, detail)

    # max_lifetime wins over everything except an active lock. Measured per
    # session: a restart is a deliberate new lease, not a forgotten instance.
    if cfg.max_lifetime_s is not None and snapshot.session_age_seconds() > cfg.max_lifetime_s:
        if locked:
            return Decision(ACTION_NONE, "locked", detail)
        action = _terminal_action(cfg)
        if action == ACTION_NONE:
            return Decision(ACTION_NONE, "max_lifetime_no_action", detail)
        return Decision(action, REASON_MAX_LIFETIME, detail)

    # Session budget ceiling.
    if cfg.max_session_cost_usd is not None and cfg.hourly_cost_usd:
        spent = cfg.hourly_cost_usd * snapshot.session_age_seconds() / 3600.0
        detail["session_cost_usd"] = round(spent, 4)
        if spent > cfg.max_session_cost_usd:
            if locked:
                return Decision(ACTION_NONE, "locked", detail)
            action = _terminal_action(cfg)
            if action != ACTION_NONE:
                return Decision(action, REASON_BUDGET, detail)

    if locked:
        return Decision(ACTION_NONE, "locked", detail)

    # Useful work in flight is never interrupted (spec criterion 5).
    if snapshot.starting:
        return Decision(ACTION_NONE, "startup_in_progress", detail)
    if snapshot.pending(cfg):
        return Decision(ACTION_NONE, "workload_active", detail)

    if cfg.idle_timeout_s is not None and snapshot.idle_seconds(cfg) > cfg.idle_timeout_s:
        action = _terminal_action(cfg)
        if action == ACTION_NONE:
            return Decision(ACTION_NONE, "idle_no_action", detail)
        return Decision(action, REASON_IDLE, detail)

    return Decision(ACTION_NONE, "within_policy", detail)
