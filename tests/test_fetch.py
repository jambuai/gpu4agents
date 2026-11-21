"""The weight fetcher (watchdog/fetch_ext.py) and how setup uses it."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fakes import FakeProvider

from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.engine import Engine
from gpu4agents.core.errors import HostFailure
from gpu4agents.runtime.base import FETCH_MARKER, FETCH_WAIT
from gpu4agents.runtime.llama import LlamaRuntime
from gpu4agents.runtime.vllm import VllmRuntime
from gpu4agents.watchdog import fetch_ext

# -- what to fetch ------------------------------------------------------------


def _files(*pairs):
    return [(p, s) for p, s in pairs]


def test_vllm_takes_the_shards_its_index_names_and_the_small_side_files():
    files = _files(
        ("config.json", 600),
        ("tokenizer.json", 7_000_000),
        ("model.safetensors.index.json", 30_000),
        ("model-00001-of-00002.safetensors", 5_000_000_000),
        ("model-00002-of-00002.safetensors", 4_000_000_000),
        ("consolidated.safetensors", 9_000_000_000),  # Mistral's second copy
        ("original/consolidated.00.pth", 9_000_000_000),
        ("samples/demo.mp4", 900_000_000),
    )
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
    paths, ref = fetch_ext.fetch_select(files, "vllm", read_json=lambda path: index)
    assert ref == ""
    assert sorted(paths) == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
    ]


def test_vllm_falls_back_to_bin_weights_and_leaves_a_repo_without_weights_to_vllm():
    paths, _ = fetch_ext.fetch_select(_files(("config.json", 1), ("pytorch_model.bin", 9)), "vllm")
    assert "pytorch_model.bin" in paths
    paths, reason = fetch_ext.fetch_select(_files(("config.json", 1), ("weights/model.safetensors", 9)), "vllm")
    assert paths is None and "repo root" in reason


def test_llama_picks_the_quant_and_every_part_of_a_split_gguf():
    files = _files(
        ("Model-UD-Q5_K_S.gguf", 24),
        ("Model-UD-Q5_K_XL.gguf", 26),
        ("big/Model-Q8_0-00001-of-00002.gguf", 50),
        ("big/Model-Q8_0-00002-of-00002.gguf", 50),
    )
    assert fetch_ext.fetch_select(files, "llama", "Q5_K_XL") == (["Model-UD-Q5_K_XL.gguf"], "Model-UD-Q5_K_XL.gguf")
    paths, ref = fetch_ext.fetch_select(files, "llama", "q8_0")
    assert paths == ["big/Model-Q8_0-00001-of-00002.gguf", "big/Model-Q8_0-00002-of-00002.gguf"]
    assert ref == "big/Model-Q8_0-00001-of-00002.gguf"


def test_llama_brings_the_one_vision_projector_along():
    files = _files(
        ("m-UD-Q5_K_XL.gguf", 26), ("mmproj-BF16.gguf", 1), ("m.imatrix.gguf", 1), ("mtp-m.gguf", 1)
    )
    paths, ref = fetch_ext.fetch_select(files, "llama", "Q5_K_XL")
    assert paths == ["m-UD-Q5_K_XL.gguf", "mmproj-BF16.gguf"]
    assert ref == "m-UD-Q5_K_XL.gguf --mmproj {dir}/mmproj-BF16.gguf"


def test_llama_leaves_ambiguous_and_multimodal_repos_to_llama_cpp():
    files = _files(("m-Q5_K_S.gguf", 1), ("m-Q5_K_XL.gguf", 1))
    paths, reason = fetch_ext.fetch_select(files, "llama", "Q5_K")
    assert paths is None and "2 GGUF files" in reason
    paths, reason = fetch_ext.fetch_select(
        files + _files(("mmproj-F16.gguf", 1), ("mmproj-F32.gguf", 1)), "llama", "Q5_K_XL"
    )
    assert paths is None and "mmproj" in reason
    # No quant: llama.cpp's own default.
    assert fetch_ext.fetch_select(_files(("m-Q4_K_M.gguf", 1), ("m-Q8_0.gguf", 1)), "llama")[1] == "m-Q4_K_M.gguf"


# -- download -----------------------------------------------------------------


class _Hub:
    """A Hugging Face stand-in: /resolve redirects to a CDN on another port."""

    def __init__(self, files, fail_first=0):
        self.files = files
        self.ranges = []
        self.auth_seen_by_cdn = []
        self.fail_first = fail_first
        hub = self

        class Cdn(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):  # noqa: N802
                name = self.path.split("/cdn/", 1)[1]
                data = hub.files[name]
                hub.auth_seen_by_cdn.append(self.headers.get("Authorization"))
                if hub.fail_first > 0:
                    hub.fail_first -= 1
                    self.send_response(500)
                    self.end_headers()
                    return
                start, end = 0, len(data) - 1
                header = self.headers.get("Range")
                if header:
                    a, b = header.split("=", 1)[1].split("-")
                    start, end = int(a), min(int(b), len(data) - 1)
                    hub.ranges.append((name, start, end))
                body = data[start : end + 1]
                self.send_response(206 if header else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class Api(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):  # noqa: N802
                if self.path.startswith("/api/models/"):
                    body = json.dumps(
                        [{"type": "file", "path": n, "size": len(d)} for n, d in hub.files.items()]
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                name = self.path.split("/resolve/main/", 1)[1]
                if name.endswith(".json"):
                    body = hub.files[name]
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{hub.cdn.server_port}/cdn/{name}")
                self.end_headers()

        self.cdn = ThreadingHTTPServer(("127.0.0.1", 0), Cdn)
        self.api = ThreadingHTTPServer(("127.0.0.1", 0), Api)
        for server in (self.cdn, self.api):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        self.endpoint = f"http://127.0.0.1:{self.api.server_port}"

    def close(self):
        for server in (self.cdn, self.api):
            server.shutdown()
            server.server_close()


@pytest.fixture
def hub(monkeypatch):
    files = {
        "config.json": b'{"model_type": "llama"}',
        "model.safetensors": bytes(range(256)) * 4000,  # 1 MB, 4 chunks of 256 KB
    }
    server = _Hub(files)
    monkeypatch.setattr(fetch_ext, "HF_ENDPOINT", server.endpoint)
    monkeypatch.setattr(fetch_ext.time, "sleep", lambda s: None)
    yield server
    server.close()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_the_download_is_byte_exact_and_the_token_never_reaches_the_cdn(hub, tmp_path):
    found = fetch_ext.fetch_plan("org/m", "main", "vllm", token="hf_secret")
    progress = fetch_ext.fetch_download(
        "org/m", "main", found["files"], str(tmp_path), "hf_secret", workers=4, chunk=256_000
    )
    data = (tmp_path / "model.safetensors").read_bytes()
    assert _sha(data) == _sha(hub.files["model.safetensors"])
    assert progress.done == found["bytes_total"]
    assert len([r for r in hub.ranges if r[0] == "model.safetensors"]) == 4
    assert set(hub.auth_seen_by_cdn) == {None}
    assert not list(tmp_path.glob("*.part*"))


def test_a_restarted_download_fetches_only_the_missing_chunks(hub, tmp_path):
    data = hub.files["model.safetensors"]
    part = tmp_path / "model.safetensors.part"
    part.write_bytes(data[:256_000] + b"\0" * (len(data) - 256_000))
    (tmp_path / "model.safetensors.part.jambu-chunks").write_text("0\n")
    fetch_ext.fetch_download("org/m", "main", [["model.safetensors", len(data)]], str(tmp_path), chunk=256_000)
    assert (tmp_path / "model.safetensors").read_bytes() == data
    assert sorted(r[1] for r in hub.ranges) == [256_000, 512_000, 768_000]


def test_a_failed_range_is_retried(hub, tmp_path):
    hub.fail_first = 2
    data = hub.files["model.safetensors"]
    fetch_ext.fetch_download("org/m", "main", [["model.safetensors", len(data)]], str(tmp_path), workers=1)
    assert (tmp_path / "model.safetensors").read_bytes() == data


# -- the watchdog side ----------------------------------------------------------


@pytest.fixture
def marker(tmp_path, monkeypatch):
    path = tmp_path / "ws" / ".jambu-fetch"
    path.parent.mkdir()
    monkeypatch.setattr(fetch_ext, "FETCH_MARKER", str(path))
    monkeypatch.setattr(fetch_ext, "FETCH_ROOT", str(tmp_path / "ws" / "models"))
    monkeypatch.setattr(fetch_ext, "_FETCH", {"state": "idle"})
    return path


def _wait_state(*states):
    deadline = time.time() + 10
    while time.time() < deadline:
        status = fetch_ext.fetch_status()
        if status["state"] in states:
            return status
        time.sleep(0.05)
    raise AssertionError(fetch_ext.fetch_status())


def test_a_finished_fetch_hands_the_launcher_a_local_path(hub, marker):
    marker.write_text("")
    code, status = fetch_ext.fetch_start({"repo": "org/m", "engine": "vllm"})
    assert code == 200 and status["state"] in ("planning", "fetching", "done")
    status = _wait_state("done", "failed")
    assert status["state"] == "done", status
    assert not marker.exists()
    local = (marker.parent / ".jambu-fetch.done").read_text().strip()
    assert local.endswith("org--m")
    assert json.loads((marker.parent / ".jambu-fetch.plan.json").read_text())["done"] is True


def test_no_fetch_when_the_launcher_is_not_waiting(hub, marker):
    _, status = fetch_ext.fetch_start({"repo": "org/m", "engine": "vllm"})
    assert status["state"] == "skipped"


def test_a_failed_fetch_lets_the_server_download_its_own_weights(hub, marker, monkeypatch):
    marker.write_text("")
    monkeypatch.setattr(fetch_ext, "FETCH_DISK_MARGIN", 1 << 60)  # "no space"
    fetch_ext.fetch_start({"repo": "org/m", "engine": "vllm"})
    status = _wait_state("failed")
    assert "free" in status["error"]
    assert not marker.exists()
    assert not (marker.parent / ".jambu-fetch.done").exists()


def test_after_a_restart_a_finished_fetch_releases_the_server_on_load(hub, marker):
    marker.write_text("")
    fetch_ext.fetch_start({"repo": "org/m", "engine": "vllm"})
    _wait_state("done")
    marker.write_text("")  # the boot script makes it again on the next start
    fetch_ext._FETCH.clear()
    fetch_ext.fetch_resume()
    assert not marker.exists()
    assert fetch_ext.fetch_status()["state"] == "done"


# -- the launcher -------------------------------------------------------------


def _wait_snippet(tmp_path):
    marker = tmp_path / ".jambu-fetch"
    script = "\n".join(FETCH_WAIT).replace(FETCH_MARKER, str(marker)) + '\necho "D=$D"\n'
    return marker, script


def test_the_launcher_waits_for_the_fetch_and_takes_its_path(tmp_path):
    marker, script = _wait_snippet(tmp_path)
    marker.write_text("fetching\n")

    def finish():
        time.sleep(1)
        (tmp_path / ".jambu-fetch.done").write_text("/workspace/models/org--m\n")
        marker.unlink()

    threading.Thread(target=finish).start()
    out = subprocess.run([shutil.which("bash"), "-c", script], capture_output=True, text=True, timeout=30)
    assert out.stdout.strip() == "D=/workspace/models/org--m"


def test_the_launcher_starts_at_once_without_a_marker(tmp_path):
    _, script = _wait_snippet(tmp_path)
    started = time.time()
    out = subprocess.run([shutil.which("bash"), "-c", script], capture_output=True, text=True, timeout=30)
    assert out.stdout.strip() == "D="
    assert time.time() - started < 2


def test_vllm_serves_the_local_path_under_the_models_own_name(write_config):
    config = write_config(runtime={"fetch": "watchdog"})
    args = VllmRuntime(config).launcher_args()
    assert f'--model "${{D:-{config.model.id}}}"' in args
    assert f"--served-model-name {config.model.id}" in args


def test_llama_switches_from_hf_to_a_local_file(write_config):
    config = write_config(
        model={"id": "org/m-GGUF"}, runtime={"engine": "llama", "quant": "Q5_K_XL"}
    )
    runtime = LlamaRuntime(config)
    assert "$F ${D:-org/m-GGUF:Q5_K_XL}" in runtime.launcher_args()
    assert 'F=-hf;[ -n "$D" ]&&F=-m' in runtime.launcher_script()
    assert runtime.fetch_request() == {
        "repo": "org/m-GGUF", "revision": "main", "engine": "llama", "quant": "Q5_K_XL"
    }


def test_auto_fetches_for_llama_and_leaves_vllm_to_huggingface_hub(write_config):
    assert VllmRuntime(write_config()).fetch_request() is None
    llama = LlamaRuntime(write_config(model={"id": "org/m-GGUF"}, runtime={"engine": "llama", "quant": "Q4_K_M"}))
    assert llama.fetch_request()["engine"] == "llama"
    assert VllmRuntime(write_config(runtime={"fetch": "watchdog"})).fetch_request()["engine"] == "vllm"


def test_fetch_server_and_no_watchdog_keep_the_old_launcher(write_config):
    for overrides in (
        {"runtime": {"fetch": "server"}},
        {"runtime": {"fetch": "watchdog"}, "lifecycle": {"watchdog": {"enabled": False}}},
    ):
        runtime = VllmRuntime(write_config(**overrides))
        assert runtime.fetch_request() is None
        assert "jambu-fetch" not in runtime.launcher_script()
        assert "jambu-fetch" not in runtime.start_command()


# -- setup ----------------------------------------------------------------------


class _Beacon:
    def __init__(self, statuses):
        self.statuses = list(statuses)

    def fetch_status(self):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]


def _engine_with(config, statuses, monkeypatch, clock):
    engine = Engine(config, provider=FakeProvider(config, CredentialResolver(config.project_dir)))
    beacon = _Beacon(statuses)
    monkeypatch.setattr(engine, "beacon", lambda state: beacon)
    monkeypatch.setattr(engine, "_instance_gone", lambda state: None)
    engine._fetching = True
    import gpu4agents.core.engine as engine_mod

    monkeypatch.setattr(engine_mod.time, "time", lambda: clock[0])

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(engine_mod.time, "sleep", sleep)
    return engine


def test_a_host_that_fetches_too_slowly_is_replaced(config, monkeypatch):
    clock = [1000.0]
    slow = {"state": "fetching", "bytes_done": 1, "bytes_total": 10**10, "rate_mbps": 50.0}
    engine = _engine_with(config, [slow], monkeypatch, clock)
    with pytest.raises(HostFailure) as caught:
        engine._await_fetch(engine.store.load())
    assert caught.value.reason == "weights download too slow on this host"
    assert clock[0] - 1000.0 >= 120


def test_a_fast_fetch_ends_the_wait_and_is_recorded(config, monkeypatch):
    clock = [1000.0]
    fast = {"state": "fetching", "bytes_done": 5, "bytes_total": 10**10, "rate_mbps": 2400.0}
    done = {"state": "done", "bytes_total": 10**10, "seconds": 40.0}
    engine = _engine_with(config, [fast] * 100 + [done], monkeypatch, clock)
    engine._await_fetch(engine.store.load())
    ended = [e for e in engine.events.tail(50) if e["event"] == "weights_fetch_ended"]
    assert ended[-1]["state"] == "done" and ended[-1]["mb_per_s"] == 250.0


def test_the_vast_offer_floor_grows_with_the_model(write_config, monkeypatch):
    from gpu4agents.providers.vast.config import VastOptions
    from gpu4agents.providers.vast.mapper import build_search_query

    config = write_config()
    engine = Engine(config, provider=FakeProvider(config, CredentialResolver(config.project_dir)))
    monkeypatch.setattr(Engine, "weights_bytes", lambda self: 55_600_000_000)
    spec = engine.build_spec()
    assert spec.min_download_mbps == 4942  # 55.6 GB in 90 s
    assert build_search_query(spec, VastOptions())["inet_down"] == {"gte": 4942.0}
    monkeypatch.setattr(Engine, "weights_bytes", lambda self: 140_000_000_000)
    assert engine.build_spec().min_download_mbps == 5000  # capped
    monkeypatch.setattr(Engine, "weights_bytes", lambda self: 1_000_000_000)
    assert build_search_query(engine.build_spec(), VastOptions())["inet_down"] == {"gte": 500.0}
    assert "inet_down" not in build_search_query(engine.build_spec(), VastOptions(min_inet_down_mbps=0))
