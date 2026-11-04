"""Serving any model behind an OpenAI-compatible endpoint.

vLLM serves chat models (runtime/vllm.py). Everything else runs as a
*service job*: a directory gpu uploads to the machine's job runner and keeps
running as the model server, on the published model port.

- ``hf/``: gpu's own server for any Hugging Face model, one task per process
  (embeddings, image, transcription, speech, chat, any other pipeline).
- ``template/``: the starting point of a server the agent writes itself
  (``runtime.engine: custom``), following SERVING.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .hf.server import PIPELINE_TAG_TASK, ROUTES, TASKS

SERVE_DIR = Path(__file__).resolve().parent
HF_SERVER_DIR = SERVE_DIR / "hf"
TEMPLATE_DIR = SERVE_DIR / "template"

# Routes vLLM's OpenAI server answers for a chat model.
VLLM_ROUTES = ["/v1/chat/completions", "/v1/completions", "/v1/responses"]

_LIBRARY_TASK = {"diffusers": "image", "sentence-transformers": "embeddings"}


def serving_for(card: dict[str, Any]) -> tuple[str, Optional[str]]:
    """(engine, task) for a Hugging Face model card (``/api/models/<repo>``).

    Chat and vision-language models go to vLLM; every other task to the hf
    server. A card with nothing to go on stays on vLLM, as before.
    """
    tag = card.get("pipeline_tag")
    library = card.get("library_name")
    task = PIPELINE_TAG_TASK.get(tag) if tag else None
    if task is None and library in _LIBRARY_TASK:
        task = _LIBRARY_TASK[library]
    if task is None and tag:
        task = "pipeline"
    if task in (None, "chat"):
        return "vllm", task
    return "hf", task


__all__ = [
    "HF_SERVER_DIR",
    "PIPELINE_TAG_TASK",
    "ROUTES",
    "TASKS",
    "TEMPLATE_DIR",
    "VLLM_ROUTES",
    "serving_for",
]
