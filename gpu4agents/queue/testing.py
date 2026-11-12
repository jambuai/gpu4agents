"""Conformance suite for QueueBackend implementations.

Subclass ``QueueBackendContract`` in a test module and provide fixtures ``backend``
and ``clock`` (an object with ``advance(seconds)`` that moves ``backend.now()``).
"""

from __future__ import annotations

import threading

import pytest

from .models import Attempt, JobState, LeaseLost, Outcome, QueueJob


def make_job(n: int, key: str = "") -> QueueJob:
    return QueueJob(
        id=f"job-{n:04d}", key=key or f"key-{n}", bundle_ref=f"bundle-{n}",
        manifest={"run": "python main.py"}, submitted_at=float(n), submitted_by="tester",
    )


class QueueBackendContract:
    def _submit(self, backend, clock, n, key=""):
        job = backend.submit(make_job(n, key))
        clock.advance(1)
        return job

    def test_submit_is_idempotent_on_key(self, backend, clock):
        a = backend.submit(make_job(1, "same"))
        b = backend.submit(make_job(2, "same"))
        assert b.id == a.id
        assert len(backend.list()) == 1

    def test_claim_returns_oldest_first(self, backend, clock):
        for n in (1, 2, 3):
            self._submit(backend, clock, n)
        assert [backend.claim("w", 60).job.id for _ in range(3)] == ["job-0001", "job-0002", "job-0003"]

    def test_a_claimed_job_is_not_claimed_again(self, backend, clock):
        self._submit(backend, clock, 1)
        assert backend.claim("w", 60) is not None
        assert backend.claim("w2", 60) is None

    def test_claim_a_specific_job(self, backend, clock):
        for n in (1, 2):
            self._submit(backend, clock, n)
        c = backend.claim("w", 60, job_id="job-0002")
        assert c is not None and c.job.id == "job-0002"
        assert backend.claim("w", 60, job_id="job-0002") is None
        assert backend.claim("w", 60, job_id="missing") is None
        # a job whose lease just expired is reaped and claimable by id
        clock.advance(61)
        again = backend.claim("w2", 60, job_id="job-0002")
        assert again is not None and again.job.id == "job-0002"

    def test_lease_expiry_requeues(self, backend, clock):
        self._submit(backend, clock, 1)
        backend.claim("w", 60)
        clock.advance(61)
        assert backend.stats().by_state.get("queued", 0) == 1
        assert backend.claim("w2", 60).job.id == "job-0001"

    def test_a_stale_token_cannot_write(self, backend, clock):
        self._submit(backend, clock, 1)
        old = backend.claim("w", 60)
        clock.advance(61)
        new = backend.claim("w2", 60)
        assert new is not None
        with pytest.raises(LeaseLost):
            backend.update(old, machine="m1")
        with pytest.raises(LeaseLost):
            backend.settle(old, JobState.SUCCEEDED, {})
        with pytest.raises(LeaseLost):
            backend.renew(old, 60)

    def test_settle_is_final(self, backend, clock):
        self._submit(backend, clock, 1)
        c = backend.claim("w", 60)
        done = backend.settle(c, JobState.SUCCEEDED, {"ok": True})
        assert done.state == JobState.SUCCEEDED
        assert backend.claim("w", 60) is None
        assert backend.claim("w", 60, job_id="job-0001") is None
        with pytest.raises(LeaseLost):
            backend.settle(c, JobState.SUCCEEDED, {})
        with pytest.raises(ValueError):
            backend.settle(c, JobState.RUNNING, {})

    def test_requeue_records_the_attempt(self, backend, clock):
        self._submit(backend, clock, 1)
        c = backend.claim("w", 60)
        att = Attempt(number=1, machine="m1", remote_id="r1", started_at=1.0,
                      ended_at=2.0, outcome=Outcome.MACHINE_LOST)
        job = backend.requeue(c, att)
        assert job.state == JobState.QUEUED
        assert [a.number for a in job.attempts] == [1]
        assert job.attempts[0].outcome == Outcome.MACHINE_LOST
        assert backend.claim("w", 60).job.attempts[0].remote_id == "r1"

    def test_cancel_of_a_queued_job_is_immediate(self, backend, clock):
        self._submit(backend, clock, 1)
        assert backend.request_cancel("job-0001", "me").state == JobState.CANCELLED
        assert backend.claim("w", 60) is None

    def test_cancel_of_a_claimed_job_sets_the_flag(self, backend, clock):
        self._submit(backend, clock, 1)
        backend.claim("w", 60)
        job = backend.request_cancel("job-0001", "me")
        assert job.cancel_requested is True
        assert job.state == JobState.CLAIMED

    def test_paused_queue_claims_nothing(self, backend, clock):
        self._submit(backend, clock, 1)
        backend.set_paused(True, "me")
        assert backend.paused() is True
        assert backend.claim("w", 60) is None
        backend.set_paused(False, "me")
        assert backend.claim("w", 60) is not None

    def test_one_coordinator_lease_at_a_time(self, backend, clock):
        assert backend.acquire_coordinator("a", 30) is True
        assert backend.acquire_coordinator("b", 30) is False
        assert backend.coordinator()["owner"] == "a"
        assert backend.renew_coordinator("a", 30) is True
        assert backend.renew_coordinator("b", 30) is False
        backend.release_coordinator("a")
        assert backend.acquire_coordinator("b", 30) is True

    def test_coordinator_lease_expires(self, backend, clock):
        assert backend.acquire_coordinator("a", 30) is True
        clock.advance(31)
        assert backend.acquire_coordinator("b", 30) is True
        assert backend.coordinator()["owner"] == "b"

    def test_stats_counts_and_oldest_age(self, backend, clock):
        assert backend.stats().oldest_queued_age_s is None
        self._submit(backend, clock, 1)
        self._submit(backend, clock, 2)
        backend.claim("w", 600)
        clock.advance(10)
        s = backend.stats()
        assert s.by_state.get("queued") == 1
        assert s.by_state.get("claimed") == 1
        assert s.oldest_queued_age_s is not None and s.oldest_queued_age_s >= 10

    def test_concurrent_claims_never_share_a_job(self, backend, clock):
        for n in range(100):
            backend.submit(make_job(n))
        got: list[str] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            for _ in range(50):
                c = backend.claim(f"w{i}", 600)
                if c is not None:
                    with lock:
                        got.append(c.job.id)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(got) == 100
        assert len(set(got)) == 100
