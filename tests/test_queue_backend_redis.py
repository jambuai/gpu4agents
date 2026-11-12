import os
import subprocess
import sys
import time
import uuid

import pytest

fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("redis")

from gpu4agents.core.errors import ConfigError  # noqa: E402
from gpu4agents.queue.backends.redis import RedisBackend, create  # noqa: E402
from gpu4agents.queue.models import JobState, LeaseLost  # noqa: E402
from gpu4agents.queue.testing import QueueBackendContract, make_job  # noqa: E402

REAL_URL = os.environ.get("GPU_TEST_REDIS_URL")


class OffsetClock:
    """Moves backend.now() by shifting the offset added to the server's TIME."""

    def __init__(self, backend):
        self.backend = backend

    def advance(self, s):
        self.backend._test_clock_offset += s


@pytest.fixture(params=["fake"] + (["real"] if REAL_URL else []))
def backend(request):
    ns = f"gpu4agents:test-{uuid.uuid4().hex[:8]}"
    if request.param == "fake":
        client = fakeredis.FakeStrictRedis(server=fakeredis.FakeServer(), decode_responses=True)
        yield RedisBackend("redis://fake", ns, client=client)
        return
    import redis
    client = redis.Redis.from_url(REAL_URL, decode_responses=True)
    be = RedisBackend(REAL_URL, ns, client=client)
    yield be
    keys = list(client.scan_iter(f"{ns}:*"))
    if keys:
        client.delete(*keys)


@pytest.fixture
def clock(backend):
    return OffsetClock(backend)


@pytest.fixture(autouse=True)
def _no_creds(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GPU_QUEUE_URL", raising=False)


class TestRedisBackend(QueueBackendContract):
    def test_lease_expiry_uses_the_server_clock(self, backend, clock, monkeypatch):
        """Meaningful only against a real Redis (GPU_TEST_REDIS_URL): fakeredis shares this
        process's clock, so the fake variant only checks the offset-driven server clock."""
        backend.submit(make_job(1))
        c = backend.claim("w", 60)
        # a client clock one hour ahead changes nothing: the server decides
        real = time.time
        if not isinstance(backend._r, fakeredis.FakeStrictRedis):
            # fakeredis runs in this process and reads time.time itself, so only a real server
            # can tell the client clock from the server clock.
            monkeypatch.setattr(time, "time", lambda: real() + 3600)
        assert backend.claim("w2", 60) is None
        assert backend.update(c, machine="m1").machine == "m1"
        monkeypatch.setattr(time, "time", real)
        clock.advance(61)
        with pytest.raises(LeaseLost):
            backend.update(c, machine="m1")
        assert backend.claim("w2", 60) is not None

    def test_now_is_the_server_time(self, backend):
        assert abs(backend.now() - time.time()) < 5

    def test_control_actions_are_audited(self, backend, clock):
        backend.submit(make_job(1))
        backend.set_paused(True, "alice")
        backend.request_cancel("job-0001", "bob")
        rows = backend.audit()
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
        got = backend.submit(job)
        assert job.submitted_at == 1.0 and got.submitted_at != 1.0

    def test_job_fields_round_trip(self, backend, clock):
        job = make_job(1)
        job.limits = {"timeout": "1h"}
        backend.submit(job)
        c = backend.claim("w", 60)
        backend.update(c, outputs_ref={"url": "s3://b/k"}, result={})
        got = backend.get("job-0001")
        assert got.limits == {"timeout": "1h"} and got.outputs_ref == {"url": "s3://b/k"}
        assert got.result == {} and got.manifest == {"run": "python main.py"}

    def test_list_is_a_snapshot_that_respects_the_filter(self, backend, clock):
        for n in range(1, 6):
            backend.submit(make_job(n))
        for _ in range(2):
            backend.claim("w", 600)
        queued = backend.list({JobState.QUEUED})
        assert [j.id for j in queued] == ["job-0003", "job-0004", "job-0005"]
        assert all(j.state == JobState.QUEUED for j in queued)
        everything = backend.list()
        assert len({j.id for j in everything}) == len(everything) == 5
        assert [j.id for j in backend.list(limit=3)] == ["job-0001", "job-0002", "job-0003"]

    def test_list_skips_and_cleans_a_stale_index_entry(self, backend, clock):
        for n in range(1, 4):
            backend.submit(make_job(n))
        ns, r = backend.namespace, backend._r
        r.delete(f"{ns}:job:job-0002")  # the hash is gone, its index entry is not
        r.zadd(f"{ns}:st:running", {"job-0003": 3.0})  # an id indexed under a state it is not in

        assert [j.id for j in backend.list({JobState.QUEUED})] == ["job-0001", "job-0003"]
        assert [(j.id, j.state) for j in backend.list()] == [
            ("job-0001", JobState.QUEUED), ("job-0003", JobState.QUEUED)
        ]
        assert r.zscore(f"{ns}:st:queued", "job-0002") is None  # cleaned, not failing every call
        assert r.zscore(f"{ns}:st:running", "job-0003") is None

    def test_submit_with_an_existing_id_and_another_key_raises(self, backend, clock):
        backend.submit(make_job(1))
        clash = make_job(1, "other-key")
        with pytest.raises(ValueError):
            backend.submit(clash)
        assert len(backend.list()) == 1
        assert backend.submit(make_job(2, "other-key")).id == "job-0002"

    def test_submit_keeps_attempts(self, backend, clock):
        from gpu4agents.queue.models import Attempt
        job = make_job(1)
        job.attempts = [Attempt(1, "m", "r", 1.0)]
        assert [a.number for a in backend.submit(job).attempts] == [1]

    def test_a_leased_job_with_a_missing_hash_is_released_not_fatal(self, backend, clock):
        backend.submit(make_job(1))
        backend.claim("w", 60)
        backend._r.delete(f"{backend.namespace}:job:job-0001")
        clock.advance(61)
        assert backend.list() == []
        assert backend.stats().by_state == {}

    def test_two_backends_share_the_namespace(self, backend):
        other = RedisBackend("redis://fake", backend.namespace, client=backend._r)
        backend.submit(make_job(1))
        assert other.claim("w", 60) is not None
        assert backend.claim("w2", 60) is None
