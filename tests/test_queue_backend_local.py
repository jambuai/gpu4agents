import pytest

from gpu4agents.queue.backends.local import LocalBackend
from gpu4agents.queue.models import JobState
from gpu4agents.queue.testing import QueueBackendContract, make_job


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def backend(tmp_path, clock):
    return LocalBackend(tmp_path / "q.db", clock=clock.now)


class TestLocalBackend(QueueBackendContract):
    def test_two_processes_share_the_file(self, tmp_path, clock):
        a = LocalBackend(tmp_path / "q.db", clock=clock.now)
        b = LocalBackend(tmp_path / "q.db", clock=clock.now)
        a.submit(make_job(1))
        assert a.claim("w", 60) is not None
        assert b.claim("w2", 60) is None
        assert b.get("job-0001").state == JobState.CLAIMED

    def test_control_actions_are_audited(self, tmp_path, clock):
        be = LocalBackend(tmp_path / "q.db", clock=clock.now)
        be.submit(make_job(1))
        be.set_paused(True, "alice")
        be.request_cancel("job-0001", "bob")
        rows = be.audit()
        assert [(r["actor"], r["action"]) for r in rows] == [("alice", "set_paused"), ("bob", "request_cancel")]

    def test_a_drain_request_is_audited(self, backend, clock):
        backend.request_drain("alice")
        assert [(r["actor"], r["action"]) for r in backend.audit()] == [("alice", "request_drain")]

    def test_update_state_only_among_leased_states(self, backend, clock):
        backend.submit(make_job(1))
        c = backend.claim("w", 60)
        assert backend.update(c, state=JobState.DELIVERING).state == JobState.DELIVERING
        for bad in (JobState.QUEUED, JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
            with pytest.raises(ValueError):
                backend.update(c, state=bad)
        with pytest.raises(ValueError):
            backend.update(c, submitted_at=1.0)
        assert backend.get("job-0001").state == JobState.DELIVERING

    def test_submit_does_not_mutate_the_callers_job(self, backend, clock):
        job = make_job(1)
        backend.submit(job)
        assert job.submitted_at == 1.0

    def test_reads_do_not_block_on_a_held_write_lock(self, tmp_path, clock):
        import sqlite3
        be = LocalBackend(tmp_path / "q.db", clock=clock.now)
        be.submit(make_job(1))
        holder = sqlite3.connect(tmp_path / "q.db", isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            assert be.get("job-0001") is not None
            assert be.stats().by_state == {"queued": 1}
            assert be.paused() is False and be.coordinator() is None and be.audit() == []
        finally:
            holder.execute("ROLLBACK")
            holder.close()
