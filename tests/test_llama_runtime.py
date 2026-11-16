"""The llama.cpp (GGUF) serving engine: config + runtime."""

from __future__ import annotations

from gpu4agents.core.config import RuntimeConfig

from conftest import BASE_CONFIG


def _cfg(**runtime):
    body = dict(BASE_CONFIG)
    body["runtime"] = {**BASE_CONFIG["runtime"], **runtime}
    return RuntimeConfig.model_validate(body)


# --- Task 1: config --------------------------------------------------------

def test_engine_accepts_llama():
    rc = _cfg(engine="llama")
    assert rc.runtime.engine == "llama"


def test_llama_quant_and_gpu_layers_parse():
    rc = _cfg(engine="llama", quant="Q5_K_XL", n_gpu_layers=40)
    assert rc.runtime.quant == "Q5_K_XL"
    assert rc.runtime.n_gpu_layers == 40


def test_llama_fields_default_empty():
    rc = _cfg(engine="llama")
    assert rc.runtime.quant is None
    assert rc.runtime.n_gpu_layers is None  # None means: offload all layers
    assert rc.runtime.llama_args == {}


# --- Task 2: LlamaRuntime --------------------------------------------------

from gpu4agents.runtime import runtime_registry  # noqa: E402


def _runtime(**runtime):
    return runtime_registry.get(_cfg(engine="llama", **runtime))


def test_llama_engine_is_registered():
    rt = _runtime(quant="Q5_K_XL")
    assert rt.engine == "llama"


def test_llama_default_image_is_pinned_not_latest():
    rt = _runtime(quant="Q5_K_XL")
    img = rt.image()
    assert "llama.cpp" in img
    assert not img.endswith(":latest") and ":" in img.rsplit("/", 1)[-1]


def test_llama_server_args_has_core_flags():
    rt = _runtime(quant="Q5_K_XL", context_length=32768, api_key="SEKRET")
    args = rt.server_args()
    joined = " ".join(args)
    # pulls the GGUF straight from HF by repo:quant
    assert "-hf" in args
    assert f"{rt.config.model.id}:Q5_K_XL" in args
    assert "--host" in args and "0.0.0.0" in args
    assert "--port" in args and str(rt.config.runtime.port) in args
    assert "--metrics" in args            # Prometheus for the watchdog probe
    assert "--jinja" in args              # model chat template -> tool calls
    assert "--ctx-size" in args and "32768" in args
    assert "--api-key" in args and "SEKRET" in args
    assert "--n-gpu-layers" in args       # offload to the GPU


def test_llama_n_gpu_layers_all_when_unset():
    # None means: offload every layer. A large number does that in llama.cpp.
    rt = _runtime(quant="Q5_K_XL")
    args = rt.server_args()
    i = args.index("--n-gpu-layers")
    assert int(args[i + 1]) >= 999


def test_llama_n_gpu_layers_explicit():
    rt = _runtime(quant="Q5_K_XL", n_gpu_layers=20)
    args = rt.server_args()
    i = args.index("--n-gpu-layers")
    assert args[i + 1] == "20"


def test_llama_llama_args_escape_hatch():
    rt = _runtime(quant="Q5_K_XL", llama_args={"flash_attn": True, "threads": 8})
    args = rt.server_args()
    assert "--flash-attn" in args
    assert "--threads" in args and "8" in args


def test_llama_activity_probe_uses_llamacpp_metrics():
    probe = _runtime(quant="Q5_K_XL").activity_probe()
    assert probe["kind"] == "prometheus"
    assert "llamacpp:requests_processing" in probe["gauges"]
    assert "llamacpp:prompt_tokens_total" in probe["counters"]


def test_llama_routes_are_openai():
    routes = _runtime(quant="Q5_K_XL").routes()
    assert "/v1/chat/completions" in routes


def test_llama_validate_requires_quant():
    result = runtime_registry.get(_cfg(engine="llama")).validate()
    assert not result.ok
    assert any("quant" in e.message.lower() for e in result.errors)


def test_llama_validate_ok_with_quant():
    assert runtime_registry.get(_cfg(engine="llama", quant="Q5_K_XL")).validate().ok


def test_llama_start_command_launches_server_and_is_bounded():
    cmd = _runtime(quant="Q5_K_XL").start_command()
    assert "llama-server" in cmd or "llama_server" in cmd or "LAUNCHER" in cmd.upper() or ".sh" in cmd
    # must fit a provider's onstart field (Vast ~16 KB)
    assert len(cmd) < 16000


def test_llama_describe_names_model_and_quant():
    text = _runtime(quant="Q5_K_XL").describe()
    assert "Q5_K_XL" in text


# --- Task 2: engine reports a chat task for llama (connection metadata) -----

def test_llama_engine_task_is_chat(write_config):
    from fakes import FakeProvider

    from gpu4agents.core.credentials import CredentialResolver
    from gpu4agents.core.engine import Engine

    cfg = write_config(runtime={"engine": "llama", "quant": "Q5_K_XL"},
                       model={"id": "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF"})
    engine = Engine(cfg, provider=FakeProvider(cfg, CredentialResolver(cfg.project_dir)))
    assert engine._task() == "chat"


# --- review fixes: boot PATH + cache dir -----------------------------------

def test_llama_launcher_invokes_resolvable_binary():
    # The image's llama-server lives at /app and /app is NOT on PATH, so a bare
    # `llama-server` is command-not-found. The launcher must resolve it.
    script = _runtime(quant="Q5_K_XL").launcher_script()
    assert "/app/llama-server" in script          # absolute fallback present
    assert "command -v llama-server" in script    # or picked up if on PATH


def test_llama_env_sets_llama_cache_dir():
    # llama.cpp's -hf downloader caches under LLAMA_CACHE, not HF_HOME.
    env = _runtime(quant="Q5_K_XL").env()
    assert env.get("LLAMA_CACHE", "").startswith("/workspace")
