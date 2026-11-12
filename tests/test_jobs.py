"""`gpu job`: code uploaded to the machine, run there, outputs brought back.

The job runner is exercised for real: a watchdog process is started from the
generated script, the CLI-side client uploads the runner to it, and jobs are
real processes. Only the GPU is missing.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path

import pytest
import yaml
from test_watchdog import policy

from gpu4agents.cli.job import shell_command
from gpu4agents.core.config import load_config
from gpu4agents.core.errors import ConfigError, StateError
from gpu4agents.core.job import (
    JobClient,
    JobManifest,
    bundle_files,
    build_bundle,
    extension_digest,
    extension_source,
    load_manifest,
    safe_extract,
)


def make_bundle(tmp_path: Path, files: dict[str, str]) -> Path:
    src = tmp_path / "src"
    for name, text in files.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_text(text)
    bundle = tmp_path / "bundle.tar.gz"
    build_bundle(src, [], bundle)
    return bundle


def wait_done(client: JobClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = client.info(job_id)
        if info["state"] != "running":
            return info
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


def full_log(client: JobClient, job_id: str) -> str:
    text, offset = "", 0
    while True:
        chunk = client.log(job_id, offset)
        text += chunk["data"]
        if chunk["offset"] == offset:
            return text
        offset = chunk["offset"]


# -- the runner on the machine -------------------------------------------------


def test_a_job_runs_its_code_and_its_outputs_come_back(runner, tmp_path):
    agent, client = runner()
    bundle = make_bundle(
        tmp_path,
        {
            "main.py": (
                "import os, pathlib\n"
                "out = pathlib.Path(os.environ['JAMBU_OUTPUT_DIR'])\n"
                "(out / 'result.txt').write_text(open('data/in.txt').read().upper())\n"
                "(out / 'sub').mkdir()\n"
                "(out / 'sub' / 'n.json').write_text('{\"n\": 3}')\n"
                "print('job says', os.environ['GREETING'], os.getcwd().endswith('src'))\n"
            ),
            "data/in.txt": "hello gpu",
        },
    )
    client.upload("j1", bundle)
    started = client.start(
        "j1",
        {"command": "python3 main.py", "setup": "echo installing", "env": {"GREETING": "hi"}, "label": "t"},
    )
    assert started["job"]["state"] == "running"

    info = wait_done(client, "j1")
    assert info["state"] == "succeeded" and info["returncode"] == 0
    assert {o["path"] for o in info["outputs"]} == {"result.txt", "sub/n.json"}
    log = full_log(client, "j1")
    assert "installing" in log and "job says hi True" in log

    files = client.download_outputs("j1", tmp_path / "back")
    assert files == ["result.txt", "sub/n.json"]
    assert (tmp_path / "back" / "result.txt").read_text() == "HELLO GPU"


def test_the_runner_is_uploaded_once_per_version(runner):
    agent, client = runner()
    digest = extension_digest(extension_source())
    assert agent.client.state()["extensions"] == {"jobs": digest}
    client.ensure_runner()
    assert len(agent.records("extension_loaded")) == 1


def test_a_failing_job_reports_its_exit_code(runner):
    _, client = runner()
    client.start("bad", {"command": "echo before; exit 3"})
    info = wait_done(client, "bad")
    assert info["state"] == "failed" and info["returncode"] == 3
    assert "before" in full_log(client, "bad")


def test_a_failed_setup_skips_the_command(runner):
    _, client = runner()
    client.start("s", {"setup": "exit 7", "command": "echo SHOULD-NOT-RUN"})
    info = wait_done(client, "s")
    assert info["returncode"] == 7
    assert "SHOULD-NOT-RUN" not in full_log(client, "s")


def test_a_running_job_holds_the_machine_and_its_end_starts_the_idle_clock(runner):
    agent, client = runner(policy(idle_timeout_s=2.0, heartbeat_grace_s=2.0), interval_s=0.5)
    client.start("long", {"command": "sleep 6"})
    time.sleep(5)
    assert agent.wait_for_marker(timeout=0.1) is None, "the guard stopped a machine running a job"
    assert agent.client.state()["decision"]["reason"] == "workload_active"

    wait_done(client, "long")
    marker = agent.wait_for_marker(timeout=15)
    assert marker and marker["action"] == "stop", "an idle machine must stop once the job ends"


def test_timeout_and_cancel_stop_the_whole_process_group(runner):
    _, client = runner()
    client.start("slow", {"command": "sleep 60 & sleep 60; wait", "timeout_s": 1})
    assert wait_done(client, "slow")["state"] == "timeout"

    client.start("c", {"command": "trap '' TERM; sleep 60"})
    time.sleep(0.5)
    assert client.cancel("c")["state"] == "cancelling"
    info = wait_done(client, "c", timeout=15)  # TERM ignored: SIGKILL after the grace
    assert info["state"] == "cancelled"


def test_the_provider_credential_does_not_reach_the_job(runner, tmp_path):
    agent, client = runner()
    agent.close()
    config = json.loads((agent.dir / "watchdog.json").read_text())
    config["env"] = {"PROVIDER_SECRET": "s3cret"}
    (agent.dir / "watchdog.json").write_text(json.dumps(config))
    agent.start()
    client.start("env", {"command": "echo secret=${PROVIDER_SECRET:-none} out=$JAMBU_OUTPUT_DIR"})
    wait_done(client, "env")
    log = full_log(client, "env")
    assert "secret=none" in log
    assert "out=" + str(agent.dir / "jobs" / "env" / "out") in log


def test_a_respawned_watchdog_reloads_the_runner_and_keeps_watching(runner):
    agent, client = runner()
    client.start("survivor", {"command": "sleep 2; echo done; exit 5"})
    agent.process.kill()
    agent.process.wait()
    agent.start()  # the supervisor's respawn, same session

    assert agent.client.state()["extensions"].get("jobs")
    info = wait_done(client, "survivor")
    assert info["state"] == "failed" and info["returncode"] == 5, "rc comes from the job's own shell"


def test_a_bundle_escaping_its_directory_is_refused(runner, tmp_path):
    _, client = runner()
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        data = b"pwned"
        member = tarfile.TarInfo("../../escaped.txt")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    client.upload("evil", evil)
    with pytest.raises(Exception, match="unpack"):
        client.start("evil", {"command": "true"})
    assert not list(tmp_path.rglob("escaped.txt"))


def test_job_ids_are_unique_and_validated(runner, tmp_path):
    _, client = runner()
    client.start("once", {"command": "true"})
    with pytest.raises(Exception, match="already started"):
        client.start("once", {"command": "true"})
    with pytest.raises(Exception, match="bad job path"):
        client.info("-starts-with-a-dash")
    with pytest.raises(StateError, match="no job"):
        client.info("missing")


def test_the_log_pages_on_line_boundaries(runner):
    _, client = runner()
    client.start("chatty", {"command": "for i in $(seq 1 500); do echo \"line $i é\"; done"})
    wait_done(client, "chatty")
    chunk = client.log("chatty", 0, limit=1000)
    assert chunk["data"].endswith("\n") and chunk["offset"] < chunk["size"]
    assert full_log(client, "chatty").splitlines()[-1] == "line 500 é"


def test_outputs_can_be_pulled_while_the_job_runs(runner, tmp_path):
    _, client = runner()
    client.start("ckpt", {"command": "echo 1 > $JAMBU_OUTPUT_DIR/step1.txt; sleep 30"})
    time.sleep(1)
    assert client.download_outputs("ckpt", tmp_path / "early") == ["step1.txt"]
    client.cancel("ckpt")


# -- the client side ---------------------------------------------------------


def test_the_bundle_leaves_out_vcs_envs_secrets_and_old_outputs(tmp_path):
    root = tmp_path / "proj"
    for name in (
        "train.py",
        "pkg/model.py",
        ".git/config",
        ".venv/bin/python",
        "pkg/__pycache__/x.pyc",
        ".env",
        ".env.local",
        "gpu-outputs/j1/a.txt",
        "data/raw/big.bin",
        "weights.ckpt",
        ".config/keep.toml",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x")
    names = sorted(rel for _, rel in bundle_files(root, ["data/raw", "*.ckpt"]))
    assert names == [".config/keep.toml", "pkg/model.py", "train.py"]


def test_the_manifest_declares_the_job(tmp_path):
    (tmp_path / "gpujob.yaml").write_text(
        yaml.safe_dump(
            {
                "run": ["python", "train.py", "--name", "a b"],
                "setup": "pip install -r requirements.txt",
                "env": {"EPOCHS": 3},
                "timeout": "2h",
                "exclude": ["data/"],
            }
        )
    )
    manifest = load_manifest(tmp_path)
    assert manifest.run == "python train.py --name 'a b'"
    assert manifest.env == {"EPOCHS": "3"}
    assert load_manifest(tmp_path / "nowhere") == JobManifest()

    (tmp_path / "gpujob.yaml").write_text("run: x\ngpu: a100\n")
    with pytest.raises(ConfigError, match="gpu"):
        load_manifest(tmp_path)


def test_one_argument_is_a_script_and_several_are_an_argv():
    assert shell_command(["pip install x && python a.py"]) == "pip install x && python a.py"
    assert shell_command(["python", "-c", "print('a b')"]) == "python -c 'print('\"'\"'a b'\"'\"')'"
    assert shell_command(None) is None


def test_outputs_never_land_outside_their_directory(tmp_path):
    archive = tmp_path / "out.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(StateError, match="outside"):
        safe_extract(archive, tmp_path / "dest")


def test_a_jobs_machine_needs_no_model(write_config):
    config = write_config(runtime={"engine": "none"}, model={"id": ""})
    assert config.model.id == ""
    with pytest.raises(ConfigError, match="model.id is required"):
        write_config(runtime={"engine": "vllm"}, model={"id": ""})


def test_init_jobs_writes_a_loadable_bare_machine(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from gpu4agents.cli.main import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["init", "--jobs", "--min-vram-gb", "48"])
    assert result.exit_code == 0, result.output
    config = load_config(tmp_path / "gpu.yaml")
    assert config.runtime.engine == "none"
    assert config.compute.gpu.min_vram_gb == 48
    assert config.compute.image.startswith("pytorch/pytorch:")


def test_the_default_jobs_image_is_never_placed_on_a_gpu_too_old_for_it(write_config):
    from gpu4agents.runtime import runtime_registry

    config = write_config(runtime={"engine": "none"}, model={"id": ""})
    runtime = runtime_registry.get(config)
    assert runtime.required_cuda() == 12.8
    assert runtime.required_compute_cap() == 7.5
    config.compute.image = "nvidia/cuda:12.8.0-runtime-ubuntu24.04"
    assert runtime_registry.get(config).required_compute_cap() is None
