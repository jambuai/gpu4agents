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
