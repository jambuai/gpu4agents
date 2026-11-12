import hashlib
import json
import subprocess
import sys

import pytest

from gpu4agents.core.config import QueueConfig, RuntimeConfig
from gpu4agents.core.errors import ConfigError
from gpu4agents.queue.outputs import LocalOutputStore, OutputRef, S3OutputStore, load_output_store


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("hello")
    (d / "sub" / "b.bin").write_bytes(b"\x00\x01" * 100)
    return d


def test_local_put_and_verify_round_trip(tmp_path, src):
    store = LocalOutputStore(tmp_path / "gpu-outputs")
    ref = store.put_outputs("job1", 1, src)
    assert store.verify(ref) == []
    assert (tmp_path / "gpu-outputs" / "job1" / "a.txt").read_text() == "hello"
    prev = tmp_path / "prev"
    store.fetch_previous(ref, prev)
    assert (prev / "sub" / "b.bin").read_bytes() == b"\x00\x01" * 100


def test_a_truncated_upload_fails_verification(tmp_path, src):
    store = LocalOutputStore(tmp_path / "gpu-outputs")
    ref = store.put_outputs("job1", 1, src)
    (tmp_path / "gpu-outputs" / "job1" / "sub" / "b.bin").write_bytes(b"\x00")
    problems = store.verify(ref)
    assert len(problems) == 1 and "sub/b.bin" in problems[0]
    (tmp_path / "gpu-outputs" / "job1" / "a.txt").unlink()
    assert any("a.txt" in p for p in store.verify(ref))


def test_manifest_has_sha256_per_file(tmp_path, src):
    store = LocalOutputStore(tmp_path / "gpu-outputs")
    ref = store.put_outputs("job1", 1, src)
    manifest = json.loads((tmp_path / "gpu-outputs" / "job1" / "MANIFEST.json").read_text())
    by_path = {f["path"]: f for f in manifest["files"]}
    assert by_path["a.txt"]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert by_path["a.txt"]["bytes"] == 5
    assert {f["path"] for f in ref.files} == {"a.txt", "sub/b.bin"}


def test_failed_attempt_goes_to_attempt_dir_and_latest_success_wins(tmp_path, src):
    store = LocalOutputStore(tmp_path / "gpu-outputs")
    ref = store.put_outputs("job1", 1, src, success=False)
    assert ".attempt-1" in ref.location
    assert (tmp_path / "gpu-outputs" / "job1" / ".attempt-1" / "a.txt").exists()
    (src / "a.txt").write_text("v2")
    ref2 = store.put_outputs("job1", 2, src)
    assert store.verify(ref2) == [] and store.verify(ref) == []
    assert (tmp_path / "gpu-outputs" / "job1" / "a.txt").read_text() == "v2"


def test_bundle_round_trip(tmp_path):
    store = LocalOutputStore(tmp_path / "gpu-outputs", bundles_dir=tmp_path / "bundles")
    bundle = tmp_path / "b.tar.gz"
    bundle.write_bytes(b"tarball")
    ref = store.put_bundle("job1", bundle)
    assert (tmp_path / "bundles" / "job1.tar.gz").read_bytes() == b"tarball"
    out = store.get_bundle(ref, tmp_path / "dl" / "x.tar.gz")
    assert out.read_bytes() == b"tarball"


def test_paths_cannot_escape_the_root(tmp_path, src):
    store = LocalOutputStore(tmp_path / "gpu-outputs")
    for bad in ("../x", "a/../../x", "/etc", ""):
        with pytest.raises(ValueError):
            store.put_outputs(bad, 1, src)
    ref = OutputRef(location=str(tmp_path / "gpu-outputs" / "job1"), files=[{"path": "../../evil", "bytes": 1, "sha256": "0"}])
    assert store.verify(ref)
    with pytest.raises(ValueError):
        store.fetch_previous(ref, tmp_path / "prev")


@pytest.fixture
def s3(monkeypatch):
    moto = pytest.importorskip("moto")
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    with moto.mock_aws():
        import boto3

        boto3.client("s3").create_bucket(Bucket="bkt")
        yield boto3.client("s3")


def test_s3_store_round_trip(tmp_path, src, s3):
    store = S3OutputStore("s3://bkt/pre")
    ref = store.put_outputs("job1", 1, src)
    assert ref.location.startswith("s3://bkt/pre/job1")
    assert store.verify(ref) == []
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket="bkt")["Contents"]}
    assert "pre/job1/MANIFEST.json" in keys and "pre/job1/a.txt" in keys
    prev = tmp_path / "prev"
    store.fetch_previous(ref, prev)
    assert (prev / "a.txt").read_text() == "hello"
    s3.put_object(Bucket="bkt", Key="pre/job1/a.txt", Body=b"he")
    assert any("a.txt" in p for p in store.verify(ref))
    b = tmp_path / "b.tar.gz"
    b.write_bytes(b"tar")
    bref = store.put_bundle("job1", b)
    assert store.get_bundle(bref, tmp_path / "o.tgz").read_bytes() == b"tar"


def test_load_output_store(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GPU_OUTPUTS_URL", raising=False)
    cfg = RuntimeConfig(runtime={"engine": "none"}, queue=QueueConfig())
    cfg.source_path = tmp_path / "proj" / "gpu.yaml"
    assert isinstance(load_output_store("local", cfg), LocalOutputStore)
    with pytest.raises(ConfigError, match="GPU_OUTPUTS_URL"):
        load_output_store("s3", cfg)
    with pytest.raises(ConfigError):
        load_output_store("nonsense", cfg)
    with pytest.raises(ConfigError):
        load_output_store("no_such_mod:f", cfg)


def test_importing_gpu4agents_without_boto3_works():
    code = (
        "import sys; sys.modules['boto3'] = None\n"
        "import gpu4agents, gpu4agents.queue.outputs as o\n"
        "from gpu4agents.core.errors import ConfigError\n"
        "import os; os.environ['GPU_OUTPUTS_URL']='s3://b/p'\n"
        "try:\n    o.S3OutputStore('s3://b/p')\nexcept ConfigError:\n    print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
