"""The queue backend contract. Leases and ages use ``now()``, never the caller's clock."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..core.errors import ConfigError
from .models import Attempt, Claim, JobState, QueueJob, QueueStats


class QueueBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def now(self) -> float: ...

    @abstractmethod
    def submit(self, job: QueueJob) -> QueueJob:
        """Idempotent on ``job.key``: returns the existing job when the key is known.

        A new job is stamped ``submitted_at = self.now()`` (the backend clock, not the caller's).
        """

    @abstractmethod
    def claim(self, worker: str, lease_s: float, job_id: Optional[str] = None) -> Optional[Claim]:
        """Oldest QUEUED job (None while the queue is paused; skips cancel_requested jobs).

        With ``job_id``, claim exactly that job if it is QUEUED (including one whose
        lease just expired and was reaped), else None. This works while the queue is
        paused too: a coordinator re-adopting a run already on a machine is not new work. A reaped claim whose job has
        ``cancel_requested`` becomes CANCELLED instead of QUEUED.
        """

    @abstractmethod
    def renew(self, claim: Claim, lease_s: float) -> Claim:
        """Raises LeaseLost on a stale token."""

    @abstractmethod
    def update(self, claim: Claim, **fields: Any) -> QueueJob:
        """Raises LeaseLost on a stale token."""

    @abstractmethod
    def requeue(self, claim: Claim, attempt: Attempt) -> QueueJob:
        """Raises LeaseLost on a stale token."""

    @abstractmethod
    def settle(self, claim: Claim, state: JobState, result: dict) -> QueueJob:
        """Final. Check order: a non-final ``state`` raises ValueError first, then a
        stale or already-settled claim raises LeaseLost."""

    @abstractmethod
    def request_cancel(self, job_id: str, actor: str) -> QueueJob: ...

    @abstractmethod
    def retry(self, job_id: str, actor: str) -> QueueJob:
        """FAILED -> QUEUED, keeping attempts. ValueError for a job that is not FAILED."""

    @abstractmethod
    def get(self, job_id: str) -> Optional[QueueJob]: ...

    @abstractmethod
    def list(self, states: Optional[set[JobState]] = None, limit: int = 100) -> list[QueueJob]: ...

    @abstractmethod
    def stats(self) -> QueueStats: ...

    @abstractmethod
    def acquire_coordinator(self, owner: str, lease_s: float) -> bool: ...

    @abstractmethod
    def renew_coordinator(self, owner: str, lease_s: float) -> bool: ...

    @abstractmethod
    def release_coordinator(self, owner: str) -> None: ...

    @abstractmethod
    def coordinator(self) -> Optional[dict]:
        """``{"owner": str, "lease_until": float}`` or None."""

    @abstractmethod
    def set_paused(self, paused: bool, actor: str) -> None: ...

    @abstractmethod
    def paused(self) -> bool: ...

    @abstractmethod
    def request_drain(self, actor: str) -> None:
        """Ask the coordinator to drain now (pause, requeue running jobs, retire every machine).
        Audited. The coordinator that holds the lease performs it on its next tick."""

    @abstractmethod
    def drain_requested(self) -> bool: ...

    @abstractmethod
    def clear_drain(self) -> None: ...


def load_backend(spec: str, config: Any) -> QueueBackend:
    """Resolve "local", "redis" or "module:factory" (the factory receives the config).

    "local" -> gpu4agents.queue.backends.local, "redis" -> gpu4agents.queue.backends.redis;
    each module exposes ``create(config: RuntimeConfig) -> QueueBackend``.
    """
    if spec in ("local", "redis"):
        module_name = "gpu4agents.queue.backends.local" if spec == "local" else "gpu4agents.queue.backends.redis"
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ConfigError(f"queue backend {spec!r} is not available: {exc}") from exc
        return module.create(config)
    if ":" not in spec:
        raise ConfigError(f"unknown queue backend {spec!r}; use local, redis or module:factory")
    module_name, _, attr = spec.partition(":")
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot load queue backend {spec!r}: {exc}") from exc
    return factory(config)
