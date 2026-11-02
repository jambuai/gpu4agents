"""model.id -> vLLM ``--tool-call-parser`` heuristics.

Lives in ``core`` (not ``runtime``) because both the vLLM runtime and
``core/model_inspect.py`` need it, and ``core`` must not depend on
``runtime`` (Provider != Runtime != Core stays a one-way dependency).

Source: https://docs.vllm.ai/en/latest/features/tool_calling/ - keep this in
sync when vLLM adds parsers; override with runtime.tool_call_parser for
anything not recognized here (fine-tunes, quantized repos, etc.).
"""

from __future__ import annotations

_TOOL_PARSER_HINTS: tuple[tuple[str, str], ...] = (
    ("llama-4", "llama4_pythonic"),
    ("llama4", "llama4_pythonic"),
    ("llama-3", "llama3_json"),
    ("llama3", "llama3_json"),
    ("hermes", "hermes"),
    ("qwen2.5", "hermes"),
    ("qwen-2.5", "hermes"),
    ("qwen3-coder", "qwen3_xml"),
    ("qwen3_coder", "qwen3_xml"),
    # Qwen3.5/3.6/3.8 (hybrid Gated-DeltaNet architecture, vLLM >=0.17) emit
    # tool calls in the SAME XML function/parameter format as Qwen3-Coder, not
    # Hermes' JSON-in-XML format - confirmed empirically on
    # empero-ai/Qwythos-9B-v2 (Qwen3.5-based): with --tool-call-parser hermes
    # the model's <tool_call><function=x><parameter=y>...</tool_call> output
    # landed as inert text in `content`; switching to qwen3_xml parses it into
    # a real `tool_calls` array. Applies to this whole "point release" Qwen3.x
    # family, unlike the original dash-versioned Qwen3 (e.g. Qwen3-8B), which
    # does use hermes per Qwen's own docs.
    ("qwen3.8", "qwen3_xml"),
    ("qwen3_8", "qwen3_xml"),
    ("qwen3.6", "qwen3_xml"),
    ("qwen3_6", "qwen3_xml"),
    ("qwen3.5", "qwen3_xml"),
    ("qwen3_5", "qwen3_xml"),
    ("qwen3", "hermes"),  # original dash-versioned Qwen3 (e.g. Qwen3-8B): hermes
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("granite-4", "granite4"),
    ("granite4", "granite4"),
    ("granite", "granite"),
    ("internlm", "internlm"),
    ("jamba", "jamba"),
    ("xlam", "xlam"),
    ("toolace", "pythonic"),
    ("deepseek-v3.1", "deepseek_v31"),
    ("deepseek-v3", "deepseek_v3"),
    ("glm-4.5", "glm45"),
    ("glm-4.7", "glm47"),
    ("functiongemma", "functiongemma"),
)


def guess_tool_call_parser(model_id: str) -> "str | None":
    """Best-effort ``--tool-call-parser`` guess from a HuggingFace repo id."""
    lowered = model_id.lower()
    for needle, parser in _TOOL_PARSER_HINTS:
        if needle in lowered:
            return parser
    return None
