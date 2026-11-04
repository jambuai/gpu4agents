"""Where gpu.yaml's numbers come from: fetch two public HF documents,
derive a VRAM/parser suggestion, cite the sources. Mocked against the real
shapes HuggingFace returns (see the live JSON captured while building this).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from gpu4agents.core.errors import JambuError
from gpu4agents.core.model_inspect import (
    HF_API_BASE,
    HF_HUB_BASE,
    inspect_model,
)

QWEN25_API = {
    "id": "Qwen/Qwen2.5-7B-Instruct",
    "gated": False,
    "safetensors": {"parameters": {"BF16": 7615616512}, "total": 7615616512},
}
QWEN25_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 32768,
}

QWYTHOS_API = {
    "id": "empero-ai/Qwythos-9B-v2",
    "gated": False,
    "safetensors": {"parameters": {"BF16": 9653104368}, "total": 9653104368},
}
QWYTHOS_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
}

FP8_API = {
    "id": "orcarouter/Qwen3.8-27B-Uncensored-FP8",
    "gated": "auto",
    "safetensors": {
        "parameters": {"BF16": 3082220272, "F8_E4M3": 24699207680},
        "total": 27781427952,
    },
}


def _mock(repo, api_json, config_json=None, config_status=200):
    respx.get(f"{HF_API_BASE}/{repo}").mock(return_value=httpx.Response(200, json=api_json))
    url = f"{HF_HUB_BASE}/{repo}/raw/main/config.json"
    if config_status == 200:
        respx.get(url).mock(return_value=httpx.Response(200, json=config_json or {}))
    else:
        respx.get(url).mock(return_value=httpx.Response(config_status))


@respx.mock
def test_plain_bf16_model_gets_a_full_reading():
    _mock("Qwen/Qwen2.5-7B-Instruct", QWEN25_API, QWEN25_CONFIG)
    result = inspect_model("Qwen/Qwen2.5-7B-Instruct")

    assert result.total_params == 7615616512
    assert result.param_dtype == "BF16"
    assert result.bytes_per_param == 2.0
    assert result.weight_gb == pytest.approx(15.23, abs=0.01)
    assert result.suggested_min_vram_gb == 22  # ceil(15.23*1.3 + 2)
    assert result.suggested_tool_call_parser == "hermes"
    assert result.is_hybrid_attention is False
    assert result.needs_trust_remote_code is False
    assert result.gated is False
    assert result.notes == []


@respx.mock
def test_hybrid_architecture_is_flagged_and_parser_left_unguessed():
    _mock("empero-ai/Qwythos-9B-v2", QWYTHOS_API, QWYTHOS_CONFIG)
    result = inspect_model("empero-ai/Qwythos-9B-v2")

    assert result.is_hybrid_attention is True
    assert result.suggested_tool_call_parser is None
    assert any("Ampere" in note for note in result.notes)
    assert any("could not guess" in note for note in result.notes)
    snippet = result.to_yaml_snippet()
    assert "name_filter: A6000" in snippet


@respx.mock
def test_dominant_dtype_wins_for_mixed_precision_checkpoints():
    """FP8 checkpoints keep a few tensors (norms/embeddings) in bf16 - the
    estimate should use the dtype that's actually most of the weight, not
    whichever key happens to sort first.
    """
    _mock(
        "orcarouter/Qwen3.8-27B-Uncensored-FP8",
        FP8_API,
        {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"},
    )
    result = inspect_model("orcarouter/Qwen3.8-27B-Uncensored-FP8")

    assert result.param_dtype == "F8_E4M3"
    assert result.bytes_per_param == 1.0
    assert result.total_params == 27781427952
    assert result.gated == "auto" or result.gated is True or result.gated  # truthy either way


@respx.mock
def test_quantization_config_overrides_dtype_for_byte_width():
    _mock(
        "org/some-awq-model",
        {"id": "org/some-awq-model", "gated": False, "safetensors": {"total": 7_000_000_000}},
        {
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "float16",
            "quantization_config": {"quant_method": "awq"},
        },
    )
    result = inspect_model("org/some-awq-model")
    assert result.quant_method == "awq"
    assert result.bytes_per_param == 0.5  # awq wins over the float16 torch_dtype


@respx.mock
def test_auto_map_means_trust_remote_code():
    _mock(
        "org/custom-code-model",
        {"id": "org/custom-code-model", "gated": False, "safetensors": {"total": 1_000_000_000}},
        {"architectures": ["CustomForCausalLM"], "auto_map": {"AutoModel": "modeling.CustomModel"}},
    )
    result = inspect_model("org/custom-code-model")
    assert result.needs_trust_remote_code is True
    assert "trust_remote_code: true" in result.to_yaml_snippet()


@respx.mock
def test_missing_safetensors_metadata_is_reported_not_guessed():
    _mock(
        "org/gguf-only-repo",
        {"id": "org/gguf-only-repo", "gated": False},
        {"architectures": ["SomeForCausalLM"]},
    )
    result = inspect_model("org/gguf-only-repo")
    assert result.total_params is None
    assert result.suggested_min_vram_gb is None
    assert any("no safetensors metadata" in note for note in result.notes)
    assert "UNKNOWN" in result.to_yaml_snippet()


@respx.mock
def test_config_json_absent_is_not_fatal():
    """Some repos (pure GGUF, for instance) have no config.json at the repo
    root - degrade gracefully instead of failing the whole inspection.
    """
    respx.get(f"{HF_API_BASE}/org/gguf-repo").mock(
        return_value=httpx.Response(
            200, json={"id": "org/gguf-repo", "gated": False, "safetensors": {"total": 1}}
        )
    )
    respx.get(f"{HF_HUB_BASE}/org/gguf-repo/raw/main/config.json").mock(
        return_value=httpx.Response(404)
    )
    result = inspect_model("org/gguf-repo")
    assert result.architectures == []
    assert result.model_type is None


@respx.mock
def test_unknown_repo_is_a_normalized_error():
    respx.get(f"{HF_API_BASE}/org/does-not-exist").mock(return_value=httpx.Response(404))
    with pytest.raises(JambuError):
        inspect_model("org/does-not-exist")


@respx.mock
def test_gated_repo_without_a_token_is_a_clear_error():
    respx.get(f"{HF_API_BASE}/org/gated-repo").mock(
        return_value=httpx.Response(200, json={"id": "org/gated-repo", "gated": "manual"})
    )
    respx.get(f"{HF_HUB_BASE}/org/gated-repo/raw/main/config.json").mock(
        return_value=httpx.Response(403)
    )
    with pytest.raises(JambuError) as exc:
        inspect_model("org/gated-repo")
    assert "gated" in str(exc.value).lower()


@respx.mock
def test_a_token_is_sent_when_provided():
    route = respx.get(f"{HF_API_BASE}/org/gated-repo").mock(
        return_value=httpx.Response(200, json={"id": "org/gated-repo", "gated": "manual"})
    )
    respx.get(f"{HF_HUB_BASE}/org/gated-repo/raw/main/config.json").mock(
        return_value=httpx.Response(200, json={})
    )
    inspect_model("org/gated-repo", token="hf_xxx")
    assert route.calls[0].request.headers["authorization"] == "Bearer hf_xxx"


@respx.mock
def test_a_hybrid_model_too_big_for_a6000_does_not_suggest_it():
    """Regression: name_filter: A6000 was suggested even for a 70GB-weight
    MoE model that could never fit a 48GB card.
    """
    _mock(
        "empero-ai/Qwen3.8-35B-A3B-Distill",
        {
            "id": "empero-ai/Qwen3.8-35B-A3B-Distill",
            "gated": False,
            "safetensors": {"parameters": {"BF16": 35_110_000_000}, "total": 35_110_000_000},
        },
        {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"},
    )
    result = inspect_model("empero-ai/Qwen3.8-35B-A3B-Distill")
    assert result.suggested_min_vram_gb > 44
    snippet = result.to_yaml_snippet()
    assert "name_filter: A6000" not in snippet
    assert "count: 2" in snippet  # suggested as a commented-out option
