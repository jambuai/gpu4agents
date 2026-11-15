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
