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
        self.backend._offset += s


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

    def test_two_backends_share_the_namespace(self, backend):
        other = RedisBackend("redis://fake", backend.namespace, client=backend._r)
        backend.submit(make_job(1))
        assert other.claim("w", 60) is not None
        assert backend.claim("w2", 60) is None


def test_missing_url_names_the_variable(tmp_path):
    class Cfg:
        project_dir = tmp_path

    with pytest.raises(ConfigError, match="GPU_QUEUE_URL"):
        create(Cfg())


def test_create_reads_the_url_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_QUEUE_URL", "redis://:s3cret@localhost:6399/3")

    class Cfg:
        project_dir = tmp_path / "myproj"

    be = create(Cfg())
    assert be.namespace.startswith("gpu4agents:")
    assert "s3cret" not in repr(be) and "s3cret" not in be.namespace


def test_importing_gpu4agents_without_redis_works():
    code = (
        "import sys; sys.modules['redis'] = None\n"
        "import gpu4agents, gpu4agents.queue.backend as b\n"
        "from gpu4agents.core.errors import ConfigError\n"
        "try:\n    b.load_backend('redis', None)\nexcept ConfigError as e:\n    print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
