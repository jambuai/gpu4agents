"""Where jambu.yaml's model-specific values actually come from.

Two public, documented HuggingFace endpoints - not scraping, not guessing:

    https://huggingface.co/api/models/<repo>           (HF Hub API)
        -> safetensors.total / safetensors.parameters (real on-disk param
           count, broken down by dtype), tags, pipeline_tag, gated status

    https://huggingface.co/<repo>/raw/<revision>/config.json
        -> architectures, model_type, torch_dtype, quantization_config,
           max_position_embeddings, rope_scaling, auto_map (its presence is
           the actual signal HF's own `transformers`/vLLM use to decide
           whether a repo needs --trust-remote-code)

`gpu inspect-model <repo>` fetches both, derives a VRAM estimate and a
tool-call-parser guess from them, and prints a ready `gpu_runtime:` snippet -
every number it prints traces back to one of the two documents above, and the
command says which field.

This is deliberately a heuristic, not an oracle: total_params * bytes_per_param
is the weight footprint; KV cache and activation memory depend on context
length and concurrency this module has no way to know, so a flat overhead
margin stands in for them. Hybrid/linear-attention architectures (Gated-
DeltaNet, Mamba, ...) additionally need a modern-enough GPU regardless of how
much VRAM it has - flagged separately, not folded into the VRAM number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .errors import JambuError
from .tool_parsers import guess_tool_call_parser

HF_API_BASE = "https://huggingface.co/api/models"
HF_HUB_BASE = "https://huggingface.co"

# dtype/quant-method name -> bytes per parameter. Sources: the same strings
# HF's `safetensors` API and `config.json`'s `torch_dtype` /
# `quantization_config.quant_method` use.
_BYTES_PER_DTYPE: dict[str, float] = {
    "f64": 8.0, "float64": 8.0,
    "f32": 4.0, "float32": 4.0,
    "f16": 2.0, "float16": 2.0,
    "bf16": 2.0, "bfloat16": 2.0,
    "f8_e4m3": 1.0, "f8_e5m2": 1.0, "fp8": 1.0, "float8": 1.0,
    "i8": 1.0, "int8": 1.0,
    "i4": 0.5, "int4": 0.5,
    "awq": 0.5, "gptq": 0.5, "bitsandbytes_4bit": 0.5, "nf4": 0.5,
    "nvfp4": 0.5,
}

# model_type / architectures substrings known to use hybrid linear-attention
# (Gated-DeltaNet, Mamba, ...) - these need a modern GPU (Ampere or newer)
# for their Triton/FLA kernels regardless of how much VRAM it has. Confirmed
# the hard way: a Turing card (Quadro RTX 8000) failed to boot vLLM at all
# for a qwen3_5 model; an Ampere card (RTX A6000) worked immediately.
_HYBRID_ATTENTION_MARKERS = (
    "qwen3_5", "qwen3_6", "qwen3_8", "qwen3_next",
    "jamba", "zamba", "mamba", "bamba", "recurrent_gemma",
    "falcon_mamba", "deltanet", "gdn", "minimax",
)


class ModelInspectError(JambuError):
    exit_code = 12


@dataclass
class ModelInspection:
    repo_id: str
    revision: str

    # Raw source documents, kept so callers/tests can cite exactly where a
    # number came from.
    hf_api: dict[str, Any] = field(default_factory=dict)
    hf_config: dict[str, Any] = field(default_factory=dict)

    total_params: Optional[int] = None
    param_dtype: Optional[str] = None
    quant_method: Optional[str] = None
    bytes_per_param: Optional[float] = None

    architectures: list[str] = field(default_factory=list)
    model_type: Optional[str] = None
    max_position_embeddings: Optional[int] = None
    needs_trust_remote_code: bool = False
    is_hybrid_attention: bool = False
    gated: bool = False

    suggested_tool_call_parser: Optional[str] = None

    notes: list[str] = field(default_factory=list)

    # -- derived sizing -------------------------------------------------

    @property
    def weight_gb(self) -> Optional[float]:
        if self.total_params is None or self.bytes_per_param is None:
            return None
        return self.total_params * self.bytes_per_param / 1e9

    @property
    def suggested_min_vram_gb(self) -> Optional[int]:
        """Weights + a flat headroom margin for KV cache/activations/runtime.

        Not exact - context length and concurrency change real KV-cache use
        and this function knows neither. Treat this as a floor to search
        offers from, not a guarantee; confirm with a real health check.
        """
        weight_gb = self.weight_gb
        if weight_gb is None:
            return None
        return math.ceil(weight_gb * 1.3 + 2.0)

    @property
    def suggested_context_length(self) -> int:
        cap = 32768
        if self.max_position_embeddings and self.max_position_embeddings < cap:
            return self.max_position_embeddings
        return cap

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "total_params": self.total_params,
            "param_dtype": self.param_dtype,
            "quant_method": self.quant_method,
            "bytes_per_param": self.bytes_per_param,
            "weight_gb": self.weight_gb,
            "suggested_min_vram_gb": self.suggested_min_vram_gb,
            "architectures": self.architectures,
            "model_type": self.model_type,
            "max_position_embeddings": self.max_position_embeddings,
            "suggested_context_length": self.suggested_context_length,
            "needs_trust_remote_code": self.needs_trust_remote_code,
            "is_hybrid_attention": self.is_hybrid_attention,
            "gated": self.gated,
            "suggested_tool_call_parser": self.suggested_tool_call_parser,
            "notes": self.notes,
        }

    def to_profile_fragment(self) -> dict[str, Any]:
        """The catalog-entry shape for ``jambu.models.yaml`` - just the
        model-specific fields, no lifecycle/health/budget/watchdog (those are
        project policy, not model identity, and stay in each project's own
        jambu.yaml).
        """
        model: dict[str, Any] = {"id": self.repo_id}
        if self.gated:
            model["hf_token_env"] = "HF_TOKEN"
        if self.needs_trust_remote_code:
            model["trust_remote_code"] = True

        gpu: dict[str, Any] = {"count": 1}
        if self.suggested_min_vram_gb:
            gpu["min_vram_gb"] = self.suggested_min_vram_gb
        if self.is_hybrid_attention and self.suggested_min_vram_gb and self.suggested_min_vram_gb <= 44:
            gpu["name_filter"] = "A6000"

        runtime: dict[str, Any] = {
            "engine": "vllm",
            "context_length": self.suggested_context_length,
            # auto either way: resolves the parser automatically when
            # recognized, stays off (never fights a model that doesn't do
            # tool calling) when it isn't - see docs/CHOOSING_A_MODEL.md.
            "tool_calling": "auto",
        }
        if self.quant_method:
            runtime["dtype"] = "auto"

        return {"model": model, "compute": {"gpu": gpu}, "runtime": runtime}

    def to_yaml_snippet(self) -> str:
        lines = ["gpu_runtime:", "  model:", f"    id: {self.repo_id}"]
        if self.gated:
            lines.append("    hf_token_env: HF_TOKEN  # gated repo - needs a real HF token")
        if self.needs_trust_remote_code:
            lines.append("    trust_remote_code: true  # config.json has auto_map (custom code)")
        lines.append("")
        lines.append("  compute:")
        lines.append("    gpu:")
        vram = self.suggested_min_vram_gb
        if vram:
            lines.append(f"      min_vram_gb: {vram}  # estimate, see notes")
        else:
            lines.append("      min_vram_gb: 24  # UNKNOWN - safetensors metadata unavailable, size manually")
        if self.is_hybrid_attention:
            # A6000 (48GB, Ampere) is only a safe default when the estimate
            # actually fits it - naming a card too small for a 70GB+ model
            # would just make every offer search come back empty.
            if vram and vram <= 44:
                lines.append(
                    "      name_filter: A6000  # hybrid linear-attention: needs Ampere+, not Turing"
                )
            else:
                lines.append(
                    "      # hybrid linear-attention needs Ampere+ (not Turing), but this "
                    "model is too big for a single 48GB card - either search a bigger one "
                    "(A100/H100 80GB) or split across multiple Ampere+ GPUs, e.g.:"
                )
                lines.append("      # count: 2  # + runtime.tensor_parallel_size: 2")
        lines.append("      count: 1")
        lines.append("")
        lines.append("  runtime:")
        lines.append("    engine: vllm")
        lines.append(f"    context_length: {self.suggested_context_length}")
        if self.quant_method:
            lines.append("    dtype: auto  # quantization declared in the checkpoint's own config.json")
        if self.suggested_tool_call_parser:
            lines.append(
                f"    tool_calling: auto  # resolves to {self.suggested_tool_call_parser} automatically"
            )
        else:
            lines.append("    tool_calling: auto  # gpu couldn't guess a parser for this repo name")
            lines.append("    # tool_call_parser: ???  # set explicitly if this model does tool calling")
        return "\n".join(lines) + "\n"


def _extract_dtype_and_params(api_data: dict[str, Any]) -> tuple[Optional[int], Optional[str]]:
    """From the HF Hub API's `safetensors` field - the real, on-disk shape."""
    safetensors = api_data.get("safetensors") or {}
    total = safetensors.get("total")
    per_dtype = safetensors.get("parameters") or {}
    if not per_dtype:
        return total, None
    dominant = max(per_dtype.items(), key=lambda kv: kv[1])[0]
    return total, dominant


def _bytes_per_param_for(dtype: Optional[str], quant_method: Optional[str]) -> Optional[float]:
    for candidate in (quant_method, dtype):
        if not candidate:
            continue
        key = str(candidate).lower().replace("-", "_")
        if key in _BYTES_PER_DTYPE:
            return _BYTES_PER_DTYPE[key]
    return None


def _is_hybrid_attention(model_type: Optional[str], architectures: list[str]) -> bool:
    haystack = " ".join([model_type or "", *architectures]).lower()
    return any(marker in haystack for marker in _HYBRID_ATTENTION_MARKERS)


def fetch_hf_api_metadata(
    repo_id: str, token: Optional[str] = None, timeout: float = 20.0
) -> dict[str, Any]:
    """GET https://huggingface.co/api/models/<repo_id>"""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = httpx.get(f"{HF_API_BASE}/{repo_id}", headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ModelInspectError(f"could not reach huggingface.co: {exc}") from exc
    if response.status_code == 404:
        raise ModelInspectError(f"'{repo_id}' was not found on HuggingFace")
    if response.status_code in (401, 403):
        raise ModelInspectError(
            f"'{repo_id}' is gated or private - set HF_TOKEN (a real token with access) and retry"
        )
    if response.status_code >= 400:
        raise ModelInspectError(f"HuggingFace API returned {response.status_code} for '{repo_id}'")
    return response.json()


def fetch_hf_config(
    repo_id: str, revision: str = "main", token: Optional[str] = None, timeout: float = 20.0
) -> dict[str, Any]:
    """GET https://huggingface.co/<repo_id>/raw/<revision>/config.json"""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{HF_HUB_BASE}/{repo_id}/raw/{revision}/config.json"
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ModelInspectError(f"could not reach huggingface.co: {exc}") from exc
    if response.status_code == 404:
        # Not every repo ships a config.json at the root (some are pure GGUF
        # repos, for instance) - this is not fatal, just less precise.
        return {}
    if response.status_code in (401, 403):
        raise ModelInspectError(
            f"'{repo_id}' is gated or private - set HF_TOKEN (a real token with access) and retry"
        )
    if response.status_code >= 400:
        raise ModelInspectError(f"HuggingFace returned {response.status_code} fetching config.json")
    try:
        return response.json()
    except ValueError:
        return {}


def inspect_model(
    repo_id: str, revision: str = "main", token: Optional[str] = None
) -> ModelInspection:
    api_data = fetch_hf_api_metadata(repo_id, token=token)
    config = fetch_hf_config(repo_id, revision=revision, token=token)

    total_params, dtype_from_weights = _extract_dtype_and_params(api_data)
    quant_method = None
    quant_config = config.get("quantization_config")
    if isinstance(quant_config, dict):
        quant_method = quant_config.get("quant_method")

    dtype = dtype_from_weights or config.get("torch_dtype")
    bytes_per_param = _bytes_per_param_for(dtype, quant_method)

    architectures = list(config.get("architectures") or [])
    model_type = config.get("model_type")

    result = ModelInspection(
        repo_id=repo_id,
        revision=revision,
        hf_api=api_data,
        hf_config=config,
        total_params=total_params,
        param_dtype=dtype,
        quant_method=quant_method,
        bytes_per_param=bytes_per_param,
        architectures=architectures,
        model_type=model_type,
        max_position_embeddings=config.get("max_position_embeddings"),
        needs_trust_remote_code="auto_map" in config,
        is_hybrid_attention=_is_hybrid_attention(model_type, architectures),
        gated=bool(api_data.get("gated")),
        suggested_tool_call_parser=guess_tool_call_parser(repo_id),
    )

    if total_params is None:
        result.notes.append(
            "no safetensors metadata from the HF API - this repo may be GGUF-only "
            "or not indexed; size compute.gpu.min_vram_gb manually"
        )
    if bytes_per_param is None and total_params is not None:
        result.notes.append(
            f"could not map dtype '{dtype}' to a byte width - "
            "suggested_min_vram_gb is unavailable, size manually"
        )
    if result.is_hybrid_attention:
        result.notes.append(
            "hybrid linear-attention architecture (Gated-DeltaNet/Mamba-style) - "
            "needs an Ampere-or-newer GPU for its Triton/FLA kernels; a Turing card "
            "(e.g. Quadro RTX 8000) will fail to boot regardless of VRAM"
        )
    if result.needs_trust_remote_code:
        result.notes.append("config.json has auto_map - set model.trust_remote_code: true")
    if not result.suggested_tool_call_parser:
        result.notes.append(
            "gpu could not guess a --tool-call-parser from this repo name "
            "(common for fine-tunes with custom names) - if the base model does "
            "tool calling, set runtime.tool_call_parser explicitly and verify with "
            "a live request; if unsure, leave tool_calling: auto (off) and test later"
        )
    if (
        result.max_position_embeddings
        and result.max_position_embeddings > result.suggested_context_length
    ):
        result.notes.append(
            f"native context is {result.max_position_embeddings} tokens - "
            f"suggesting {result.suggested_context_length} as a cheaper default; "
            "raise runtime.context_length deliberately if you need more"
        )

    return result
