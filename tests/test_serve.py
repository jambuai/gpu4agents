"""Serving any model: the hf server's OpenAI routes, service jobs, config and init.

The hf server's HTTP layer runs for real here with stand-in backends (no
torch on the test machine); the backends themselves are exercised live on a
GPU. Service jobs run for real on a watchdog process.
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
import threading
import time
import wave

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from gpu4agents.core.config import load_config
from gpu4agents.core.errors import ConfigError
from gpu4agents.runtime import runtime_registry
from gpu4agents.runtime.service import HF_BUNDLE_EXCLUDE, CustomRuntime, HfRuntime, ServiceSpec
from gpu4agents.serve import HF_SERVER_DIR, TEMPLATE_DIR, serving_for
from gpu4agents.serve.hf import server as hf

# --------------------------------------------------------------------------
# which engine and task a model card maps to
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "card, expected",
    [
        ({"pipeline_tag": "text-generation", "library_name": "transformers"}, ("vllm", "chat")),
        ({"pipeline_tag": "image-text-to-text"}, ("vllm", "chat")),
        ({"pipeline_tag": "text-to-image", "library_name": "diffusers"}, ("hf", "image")),
        ({"pipeline_tag": "automatic-speech-recognition"}, ("hf", "transcription")),
        ({"pipeline_tag": "text-to-speech"}, ("hf", "speech")),
        ({"pipeline_tag": "text-to-audio"}, ("hf", "speech")),
        ({"pipeline_tag": "feature-extraction"}, ("hf", "embeddings")),
        ({"pipeline_tag": "sentence-similarity"}, ("hf", "embeddings")),
        ({"library_name": "sentence-transformers"}, ("hf", "embeddings")),
        ({"library_name": "diffusers"}, ("hf", "image")),
        ({"pipeline_tag": "object-detection"}, ("hf", "pipeline")),
        ({"pipeline_tag": "translation"}, ("hf", "pipeline")),
        ({}, ("vllm", None)),
    ],
)
def test_a_model_card_picks_the_engine_and_task(card, expected):
    assert serving_for(card) == expected


# --------------------------------------------------------------------------
# the hf server's HTTP layer
# --------------------------------------------------------------------------


class FakeBackend:
    def __init__(self, task):
        self.task = task
        self.gate = threading.Event()
        self.gate.set()

    def load(self):
        self.gate.wait(10)

    def chat(self, body):
        last = body["messages"][-1]["content"]
        return f"echo: {hf.message_text(last)}", "stop", 7, 3

    def embed(self, texts):
        return [[float(len(t)), 0.5, -1.0] for t in texts]

    def count_tokens(self, texts):
        return sum(len(t.split()) for t in texts)

    def generate(self, body):
        return [("png", base64.b64encode(b"\x89PNG-fake-%d" % i).decode()) for i in range(int(body.get("n") or 1))]

    def transcribe(self, data, language=None, translate=False, prompt=None):
        self.received = (data, language, translate)
        segments = [
            {"id": 0, "start": 0.0, "end": 1.5, "text": " hello"},
            {"id": 1, "start": 1.5, "end": 3.0, "text": " world"},
        ]
        return "hello world", segments, 3.0

    def speak(self, text):
        return [0.0, 0.5, -0.5, 1.0], 16000

    def run(self, body):
        if "inputs" not in body:
            raise hf.BadRequest("inputs is required")
        return {"labels": ["cat"], "inputs": body["inputs"]}


@pytest.fixture
def serve():
    started = []

    def _start(task, api_key="sk-test", ready=True):
        env = {"JAMBU_MODEL_ID": "org/model", "JAMBU_API_KEY": api_key, "PORT": "0", "JAMBU_TASK": task}
        settings = hf.Settings(env)
        settings.pipeline_tag = "image-classification" if task == "pipeline" else None
        backend = FakeBackend(task)
        service = hf.Service(settings, task, backend)
        server = hf.make_server(service, host="127.0.0.1")
        settings.port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        if ready:
            service.load()
        started.append(server)
        base = f"http://127.0.0.1:{settings.port}"
        client = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return service, client

    yield _start
    for server in started:
        server.shutdown()


def test_health_is_503_while_loading_then_200(serve):
    service, client = serve("embeddings", ready=False)
    assert client.get("/health").status_code == 503
    service.load()
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["task"] == "embeddings"


def test_routes_need_the_api_key_but_health_and_metrics_do_not(serve):
    _, client = serve("embeddings")
    anonymous = httpx.Client(base_url=str(client.base_url), timeout=10)
    assert anonymous.get("/health").status_code == 200
    assert anonymous.get("/metrics").status_code == 200
    assert anonymous.get("/v1/models").status_code == 401
    assert anonymous.post("/v1/embeddings", json={"input": "x"}).status_code == 401
    wrong = httpx.Client(base_url=str(client.base_url), headers={"Authorization": "Bearer nope"})
    assert wrong.post("/v1/embeddings", json={"input": "x"}).status_code == 401


def test_models_lists_the_served_name_task_and_routes(serve):
    _, client = serve("image")
    data = client.get("/v1/models").json()["data"][0]
    assert data["id"] == "org/model"
    assert data["task"] == "image"
    assert data["routes"] == ["/v1/images/generations"]


def test_a_route_of_another_task_says_what_this_model_serves(serve):
    _, client = serve("speech")
    response = client.post("/v1/embeddings", json={"input": "x"})
    assert response.status_code == 404
    assert "speech" in response.json()["error"]["message"]
    assert "/v1/audio/speech" in response.json()["error"]["message"]
    assert client.post("/v1/nothing", json={}).status_code == 404


def test_metrics_count_requests_for_the_activity_probe(serve):
    _, client = serve("embeddings")
    for _ in range(3):
        client.post("/v1/embeddings", json={"input": "hello"})
    client.post("/v1/embeddings", json={"input": 5})  # a bad request still counts
    text = client.get("/metrics").text
    assert "jambu_requests_total 4" in text
    assert "jambu_request_errors_total 1" in text
    assert "jambu_requests_running 0" in text
    assert "jambu_model_ready 1" in text


def test_embeddings_in_float_and_base64(serve):
    _, client = serve("embeddings")
    body = client.post("/v1/embeddings", json={"input": ["ab", "abcd"]}).json()
    assert [d["embedding"] for d in body["data"]] == [[2.0, 0.5, -1.0], [4.0, 0.5, -1.0]]
    assert body["usage"]["prompt_tokens"] == 2
    # The openai SDK asks for base64 unless told otherwise.
    packed = client.post("/v1/embeddings", json={"input": "abc", "encoding_format": "base64"}).json()
    raw = base64.b64decode(packed["data"][0]["embedding"])
    assert list(struct.unpack("<3f", raw)) == [3.0, 0.5, -1.0]
    assert client.post("/v1/embeddings", json={"input": "abcd", "dimensions": 2}).json()["data"][0][
        "embedding"
    ] == [4.0, 0.5]


def test_images_come_back_as_b64_json_or_data_urls(serve):
    _, client = serve("image")
    body = client.post("/v1/images/generations", json={"prompt": "a cat", "n": 2}).json()
    assert len(body["data"]) == 2 and base64.b64decode(body["data"][0]["b64_json"]).startswith(b"\x89PNG")
    url = client.post("/v1/images/generations", json={"prompt": "a cat", "response_format": "url"}).json()
    assert url["data"][0]["url"].startswith("data:image/png;base64,")


def test_transcription_takes_multipart_and_every_response_format(serve):
    service, client = serve("transcription")
    files = {"file": ("talk.wav", b"RIFF-audio-bytes", "audio/wav")}
    body = client.post(
        "/v1/audio/transcriptions", files=files, data={"model": "whisper", "language": "pt"}
    ).json()
    assert body == {"text": "hello world"}
    assert service.backend.received == (b"RIFF-audio-bytes", "pt", False)

    text = client.post("/v1/audio/transcriptions", files=files, data={"response_format": "text"})
    assert text.text == "hello world\n"
    srt = client.post("/v1/audio/transcriptions", files=files, data={"response_format": "srt"}).text
    assert "1\n00:00:00,000 --> 00:00:01,500\nhello" in srt
    vtt = client.post("/v1/audio/transcriptions", files=files, data={"response_format": "vtt"}).text
    assert vtt.startswith("WEBVTT") and "00:00:01.500 --> 00:00:03.000" in vtt
    verbose = client.post(
        "/v1/audio/transcriptions", files=files, data={"response_format": "verbose_json"}
    ).json()
    assert verbose["duration"] == 3.0 and len(verbose["segments"]) == 2

    client.post("/v1/audio/translations", files=files)
    assert service.backend.received[2] is True
    missing = client.post("/v1/audio/transcriptions", files={"other": ("x", b"y")})
    assert missing.status_code == 400 and "file" in missing.json()["error"]["message"]


def test_speech_returns_a_playable_wav(serve):
    _, client = serve("speech")
    response = client.post("/v1/audio/speech", json={"input": "hello", "voice": "alloy", "response_format": "wav"})
    assert response.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(response.content)) as audio:
        assert audio.getframerate() == 16000 and audio.getnframes() == 4
    pcm = client.post("/v1/audio/speech", json={"input": "hello", "response_format": "pcm"})
    assert len(pcm.content) == 8 and pcm.headers["x-sample-rate"] == "16000"
    assert client.post("/v1/audio/speech", json={"voice": "x"}).status_code == 400


def test_chat_answers_and_streams(serve):
    _, client = serve("chat")
    body = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).json()
    assert body["choices"][0]["message"]["content"] == "echo: hi"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    streamed = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": [{"type": "text", "text": "yo"}]}]},
    ).text
    events = [line[6:] for line in streamed.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert json.loads(events[0])["choices"][0]["delta"]["content"] == "echo: yo"


def test_pipeline_runs_any_other_task(serve):
    _, client = serve("pipeline")
    body = client.post("/v1/pipeline", json={"inputs": "a photo"}).json()
    assert body["task"] == "image-classification"
    assert body["output"] == {"labels": ["cat"], "inputs": "a photo"}
    assert client.post("/v1/pipeline", json={}).status_code == 400


def test_a_request_before_the_model_loads_is_503(serve):
    service, client = serve("embeddings", ready=False)
    assert client.post("/v1/embeddings", json={"input": "x"}).status_code == 503


def test_settings_and_task_resolution_without_the_network():
    settings = hf.Settings({"JAMBU_MODEL_ID": "a/b", "JAMBU_TASK": "image", "PORT": "9000"})
    assert hf.resolve_task(settings) == "image"
    assert settings.served_name == "a/b" and settings.port == 9000
    with pytest.raises(SystemExit):
        hf.resolve_task(hf.Settings({"JAMBU_MODEL_ID": "a/b", "JAMBU_TASK": "dance"}))


def test_helpers_encode_outputs():
    with wave.open(io.BytesIO(hf.wav_bytes([0.0, 2.0, -2.0], 8000))) as audio:
        frames = audio.readframes(3)
    assert struct.unpack("<3h", frames) == (0, 32767, -32767)
    assert hf.decode_data_url("data:text/plain;base64,aGk=") == ("text/plain", b"hi")
    assert hf.decode_data_url("plain text") is None
    assert hf.jsonable({"a": (1, 2.5, float("nan")), "b": None}) == {"a": [1, 2.5, None], "b": None}
    assert hf.subtitles([{"start": 61.25, "end": 62.0, "text": " x "}], "srt").startswith(
        "1\n00:01:01,250 --> 00:01:02,000\nx"
    )


def test_the_hf_server_and_template_import_nothing_from_gpu4agents():
    # They run on the GPU machine, where gpu4agents is not installed.
    import ast

    for path in (HF_SERVER_DIR / "server.py", TEMPLATE_DIR / "server.py"):
        modules = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module or "")
        assert not any(m.startswith("gpu4agents") for m in modules), path


# --------------------------------------------------------------------------
# config and runtimes
# --------------------------------------------------------------------------


def test_hf_needs_a_model_and_custom_does_not(write_config):
    with pytest.raises(ConfigError, match="model.id is required"):
        write_config(model={"id": ""}, runtime={"engine": "hf"})
    config = write_config(model={"id": ""}, runtime={"engine": "custom"})
    assert config.serves_as_service


def test_an_unknown_task_is_refused(write_config):
    with pytest.raises(ConfigError, match="model.task"):
        write_config(model={"id": "a/b", "task": "dance"}, runtime={"engine": "hf"})


def test_the_machine_fingerprint_ignores_what_a_redeploy_changes(write_config):
    first = write_config(model={"id": "a/one", "task": "image"}, runtime={"engine": "hf"})
    second = write_config(
        model={"id": "b/two", "task": "speech"}, runtime={"engine": "hf", "packages": ["x"]}
    )
    assert first.fingerprint() == second.fingerprint()
    bigger = write_config(
        model={"id": "b/two"}, runtime={"engine": "hf"}, compute={"gpu": {"min_vram_gb": 80}}
    )
    assert bigger.fingerprint() != first.fingerprint()
    # vLLM's command line is baked into the boot script: there a model change is a new machine.
    assert write_config(model={"id": "a/one"}).fingerprint() != write_config(model={"id": "b/two"}).fingerprint()


def test_hf_runtime_shape(write_config):
    config = write_config(
        model={"id": "openai/whisper-tiny", "task": "transcription", "revision": "abc"},
        runtime={"engine": "hf", "port": 8000, "packages": ["librosa==0.11.0"]},
    )
    runtime = runtime_registry.get(config)
    assert isinstance(runtime, HfRuntime) and runtime.service
    assert runtime.ports() == [8000]
    assert runtime.start_command() == "true"
    assert runtime.api_key_env == "JAMBU_API_KEY"
    assert runtime.required_compute_cap() == 7.5  # the pytorch cu128 image
    assert runtime.routes() == ["/v1/audio/transcriptions", "/v1/audio/translations"]
    probe = runtime.activity_probe()
    assert probe["url"] == "http://127.0.0.1:8000/metrics"
    assert probe["gauges"] == ["jambu_requests_running"]
    spec = runtime.service_spec()
    assert spec.directory == HF_SERVER_DIR and spec.setup == "bash install.sh"
    assert spec.env["JAMBU_MODEL_ID"] == "openai/whisper-tiny"
    assert spec.env["JAMBU_TASK"] == "transcription"
    assert spec.env["JAMBU_REVISION"] == "abc"
    assert spec.env["JAMBU_PACKAGES"] == "librosa==0.11.0"
    assert spec.env["PORT"] == "8000"
    from gpu4agents.core.job import bundle_files

    shipped = [name for _, name in bundle_files(spec.directory, list(HF_BUNDLE_EXCLUDE))]
    assert shipped == ["install.sh", "server.py"]


def test_the_service_digest_follows_code_command_and_settings(tmp_path):
    (tmp_path / "server.py").write_text("print(1)")
    spec = ServiceSpec(tmp_path, "python3 server.py", env={"A": "1"})
    first = spec.digest()
    assert ServiceSpec(tmp_path, "python3 server.py", env={"A": "1"}).digest() == first
    assert ServiceSpec(tmp_path, "python3 server.py", env={"A": "2"}).digest() != first
    assert ServiceSpec(tmp_path, "python server.py", env={"A": "1"}).digest() != first
    (tmp_path / "server.py").write_text("print(2)")
    assert spec.digest() != first


def test_custom_runtime_reads_the_servers_manifest(write_config, tmp_path):
    config = write_config(model={"id": ""}, runtime={"engine": "custom", "service_dir": "srv"})
    runtime = runtime_registry.get(config)
    assert isinstance(runtime, CustomRuntime)
    assert runtime.validate().errors  # no srv/ yet
    srv = tmp_path / "srv"
    srv.mkdir()
    (srv / "gpujob.yaml").write_text("setup: pip install x\nenv: {MODE: fast}\n")
    with pytest.raises(ConfigError, match="no run:"):
        runtime.service_spec()
    (srv / "gpujob.yaml").write_text("run: python3 server.py\nsetup: pip install x\nenv: {MODE: fast}\n")
    spec = runtime.service_spec()
    assert spec.command == "python3 server.py" and spec.setup == "pip install x"
    assert spec.env["MODE"] == "fast" and spec.env["PORT"] == "8000"
    assert not runtime.validate().errors


# --------------------------------------------------------------------------
# gpu init
# --------------------------------------------------------------------------


def _card(monkeypatch, card):
    monkeypatch.delenv("GPU_NO_MODEL_CARD", raising=False)
    monkeypatch.setattr(
        "gpu4agents.core.model_inspect.fetch_hf_api_metadata", lambda *a, **k: card
    )


def test_init_reads_the_model_card_and_writes_an_hf_config(tmp_path, monkeypatch):
    from gpu4agents.cli.main import app

    monkeypatch.chdir(tmp_path)
    _card(monkeypatch, {"pipeline_tag": "text-to-image", "library_name": "diffusers", "safetensors": {"total": 6_000_000_000}})
    result = CliRunner().invoke(app, ["init", "--model", "org/painter"])
    assert result.exit_code == 0, result.output
    config = load_config(tmp_path / "gpu.yaml")
    assert config.runtime.engine == "hf"
    assert config.model.task == "image"
    assert config.compute.gpu.min_vram_gb == 18  # 6B params in bf16, plus headroom
    assert config.compute.image.startswith("pytorch/pytorch:")
    assert config.health.startup_timeout == "30m"


def test_init_keeps_chat_models_on_vllm(tmp_path, monkeypatch):
    from gpu4agents.cli.main import app

    monkeypatch.chdir(tmp_path)
    _card(monkeypatch, {"pipeline_tag": "text-generation", "library_name": "transformers"})
    assert CliRunner().invoke(app, ["init", "--model", "org/chat"]).exit_code == 0
    assert load_config(tmp_path / "gpu.yaml").runtime.engine == "vllm"


def test_init_with_an_unreachable_card_falls_back_to_vllm(tmp_path, monkeypatch):
    from gpu4agents.cli.main import app
    from gpu4agents.core.model_inspect import ModelInspectError

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GPU_NO_MODEL_CARD", raising=False)

    def _fail(*_a, **_k):
        raise ModelInspectError("offline")

    monkeypatch.setattr("gpu4agents.core.model_inspect.fetch_hf_api_metadata", _fail)
    result = CliRunner().invoke(app, ["init", "--model", "org/x"])
    assert result.exit_code == 0 and "assuming a chat model" in result.output
    assert load_config(tmp_path / "gpu.yaml").runtime.engine == "vllm"


def test_init_engine_and_task_flags_win(tmp_path, monkeypatch):
    from gpu4agents.cli.main import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["init", "--model", "org/x", "--engine", "hf", "--task", "speech", "--min-vram-gb", "12"])
    assert result.exit_code == 0, result.output
    config = load_config(tmp_path / "gpu.yaml")
    assert (config.runtime.engine, config.model.task, config.compute.gpu.min_vram_gb) == ("hf", "speech", 12)
    assert CliRunner().invoke(app, ["init", "--engine", "dance", "--force"]).exit_code == 2


def test_init_custom_scaffolds_a_server_that_follows_the_contract(tmp_path, monkeypatch):
    from gpu4agents.cli.main import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["init", "--engine", "custom"])
    assert result.exit_code == 0, result.output
    config = load_config(tmp_path / "gpu.yaml")
    assert config.runtime.engine == "custom" and config.model.id == ""
    assert (tmp_path / "server" / "server.py").is_file()
    assert yaml.safe_load((tmp_path / "server" / "gpujob.yaml").read_text())["run"] == "python3 server.py"
    spec = runtime_registry.get(config).service_spec()
    assert spec.directory == tmp_path / "server"
    # A second init leaves the agent's server alone.
    (tmp_path / "server" / "server.py").write_text("# mine")
    CliRunner().invoke(app, ["init", "--engine", "custom", "--force"])
    assert (tmp_path / "server" / "server.py").read_text() == "# mine"


# --------------------------------------------------------------------------
# service jobs on a live watchdog
# --------------------------------------------------------------------------


@pytest.fixture
def runner(tmp_path):
    from test_jobs import Agent, JobClient, policy

    made = []

    def _make(**overrides):
        directory = tmp_path / f"agent{len(made)}"
        directory.mkdir()
        env = {
            "JAMBU_WATCHDOG_EXT": str(directory / "ext"),
            "JAMBU_JOBS_DIR": str(directory / "jobs"),
            "JAMBU_JOB_HEARTBEAT_S": "0.3",
            "JAMBU_JOB_CANCEL_GRACE_S": "1",
        }
        agent = Agent(
            directory,
            policy(idle_timeout_s=600.0),
            env=env,
            runtime_pid_path=str(directory / "run" / "service.pid"),
            logs={"runtime": str(directory / "log" / "service.log")},
            **overrides,
        )
        made.append(agent)
        client = JobClient(agent.client)
        client.ensure_runner()
        return agent, client, directory

    yield _make
    for agent in made:
        agent.close()


def test_a_service_job_is_the_runtime_not_a_workload(runner):
    agent, client, directory = runner()
    agent.client.mark_starting()
    client.start(
        "svc1",
        {"command": "echo serving; sleep 30", "service": True, "digest": "d1", "timeout_s": 0.5},
    )
    deadline = time.time() + 10
    while time.time() < deadline and "serving" not in (directory / "log" / "service.log").read_text() if (directory / "log" / "service.log").exists() else True:
        time.sleep(0.1)
    state = agent.client.state()
    assert "svc1" not in state["workloads"], "serving must not hold the machine up by itself"
    assert state["starting"] is True, "starting a service keeps the startup grace for the CLI to end"
    pid = int((directory / "run" / "service.pid").read_text())
    assert pid == client.info("svc1")["pid"]
    assert os.path.islink(directory / "log" / "service.log")
    assert "serving" in "\n".join(agent.client.logs("runtime", 10))
    info = client.info("svc1")
    assert info["service"] is True and info["digest"] == "d1" and info["timeout_s"] is None
    time.sleep(1.5)
    assert client.info("svc1")["state"] == "running", "a service has no timeout"
    client.cancel("svc1")
    deadline = time.time() + 10
    while time.time() < deadline and client.info("svc1")["state"] == "running":
        time.sleep(0.2)
    assert client.info("svc1")["state"] == "cancelled"


def test_a_service_that_dies_while_loading_ends_the_startup_grace(runner):
    agent, client, _ = runner(startup_grace_s=600.0)
    agent.client.mark_starting()
    client.start("svc2", {"command": "echo boom >&2; exit 3", "service": True})
    deadline = time.time() + 10
    while time.time() < deadline and client.info("svc2")["state"] == "running":
        time.sleep(0.2)
    assert client.info("svc2")["returncode"] == 3
    deadline = time.time() + 10
    state = agent.client.state()
    while time.time() < deadline and state.get("starting"):
        time.sleep(0.3)
        state = agent.client.state()
    assert state["starting"] is False
    assert state["runtime_alive"] is False


def test_the_installer_upgrades_brotli_for_the_hubs_downloads():
    # Seen live: the PyTorch image's brotli 1.1 broke every weight download.
    assert "brotli>=1.2" in (HF_SERVER_DIR / "install.sh").read_text()


def test_job_list_marks_jobs_of_a_gone_machine_lost(write_config):
    # Seen live: a failed first machine's server still listed as running.
    from fakes import FakeProvider

    from gpu4agents.core.credentials import CredentialResolver
    from gpu4agents.core.engine import Engine
    from gpu4agents.core.job import JobRecords

    config = write_config()
    engine = Engine(config, provider=FakeProvider(config, CredentialResolver(config.project_dir)))
    records = JobRecords(engine.store.dir)
    records.update("j-old", state="running", instance_id="gone-1", service=True)
    records.update("j-done", state="succeeded", instance_id="gone-1")
    states = {r["id"]: r["state"] for r in engine.job_list()}
    assert states == {"j-old": "lost", "j-done": "succeeded"}
