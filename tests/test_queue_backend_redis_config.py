import subprocess
import sys

import pytest

from gpu4agents.core.config import QueueConfig, RuntimeConfig
from gpu4agents.core.errors import ConfigError


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GPU_QUEUE_URL", raising=False)
    monkeypatch.delenv("GPU_QUEUE_NAMESPACE", raising=False)


def _cfg(tmp_path, gpu="A100", **queue):
    cfg = RuntimeConfig(runtime={"engine": "none"}, compute={"disk_gb": 50 if gpu == "A100" else 60}, queue=QueueConfig(**queue))
    cfg.source_path = tmp_path / "proj" / "gpu.yaml"
    return cfg


def test_missing_url_names_the_variable(tmp_path):
    pytest.importorskip("redis")
    from gpu4agents.queue.backends.redis import create

    with pytest.raises(ConfigError, match="GPU_QUEUE_URL"):
        create(_cfg(tmp_path))


def test_namespace_resolution(tmp_path, monkeypatch):
    pytest.importorskip("redis")
    from gpu4agents.queue.backends.redis import create

    monkeypatch.setenv("GPU_QUEUE_URL", "redis://:s3cret@localhost:6399/3")
    a, b = create(_cfg(tmp_path, gpu="A100")), create(_cfg(tmp_path, gpu="H100"))
    assert a.namespace.startswith("gpu4agents:fp-") and a.namespace != b.namespace
    assert create(_cfg(tmp_path, gpu="A100")).namespace == a.namespace
    monkeypatch.setenv("GPU_QUEUE_NAMESPACE", "from-env")
    assert create(_cfg(tmp_path)).namespace == "gpu4agents:from-env"
    assert create(_cfg(tmp_path, namespace="explicit")).namespace == "gpu4agents:explicit"
    assert "s3cret" not in repr(a) and "s3cret" not in a.namespace


def test_namespace_is_validated():
    with pytest.raises(ValueError):
        QueueConfig(namespace="bad name!")
    assert QueueConfig(namespace="team-a.v1").namespace == "team-a.v1"


def test_importing_gpu4agents_without_redis_works():
    code = (
        "import sys; sys.modules['redis'] = None\n"
        "import gpu4agents, gpu4agents.queue.backend as b\n"
        "from gpu4agents.core.errors import ConfigError\n"
        "try:\n    b.load_backend('redis', None)\nexcept ConfigError:\n    print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
