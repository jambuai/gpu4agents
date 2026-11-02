"""vLLM OpenAI-compatible server runtime.

This is where "does it support agentic flows?" gets answered: vLLM's server
already implements both ``/v1/chat/completions`` (with tool/function calling)
and ``/v1/responses`` (the OpenAI Responses API, MCP-tool-routing included).
gpu does not implement any of that - it only turns on the right vLLM
flags and forwards the same endpoint/credentials it always has, so any
OpenAI-SDK-compatible agent framework (LangChain, LangGraph, CrewAI, the
`openai` SDK's own `responses.create`, ...) can talk to the instance with no
extra layer in between.
"""

from __future__ import annotations

import json
import shlex
import re
from typing import Any, Optional

from ..core.models import ValidationResult
from ..core.tool_parsers import guess_tool_call_parser  # noqa: F401 (re-exported)
from .base import ModelRuntime, cuda_from_tag, image_repo, image_tag

# vLLM flags gpu already manages explicitly - collided with via
# runtime.vllm_args, these would fight the config fields meant to control
# them (compute.gpu, model.id, lifecycle port wiring, ...). Not a hard block
# (power users can still reach them through raw extra_args if they really
# need to), just something validate() warns about.
_MANAGED_VLLM_FLAGS = frozenset(
    {
        "model",
        "host",
        "port",
        "served-model-name",
        "served_model_name",
        "revision",
        "trust-remote-code",
        "trust_remote_code",
        "enable-auto-tool-choice",
        "enable_auto_tool_choice",
        "tool-call-parser",
        "tool_call_parser",
        "tool-server",
        "tool_server",
    }
)


def flatten_vllm_args(mapping: dict[str, Any]) -> list[str]:
    """Turn a ``{flag: value}`` mapping into vLLM CLI argv.

    Rules, matched to how vLLM's own argparse-based CLI takes flags:
      - key: underscores become dashes (``max_num_seqs`` -> ``--max-num-seqs``),
        already-dashed keys pass through unchanged. A leading ``--`` in the
        key is tolerated (stripped) so either spelling works.
      - ``True``  -> a bare flag (``--enable-prefix-caching``)
      - ``False`` -> the negated flag (``--no-enable-prefix-caching``), matching
        vLLM's own store_true/store_false flag-pair convention
      - ``None``  -> the flag is skipped entirely (lets a value come from
        elsewhere, e.g. an env var, without an empty CLI arg)
      - dict/list -> JSON-encoded (several vLLM flags, e.g.
        ``--limit-mm-per-prompt``, take a JSON object as their value)
      - anything else -> ``str(value)``
    """
    args: list[str] = []
    for raw_key, value in mapping.items():
        key = raw_key.lstrip("-").replace("_", "-")
        flag = f"--{key}"
        if value is None:
            continue
        if value is True:
            args.append(flag)
        elif value is False:
            args.append(f"--no-{key}")
        elif isinstance(value, (dict, list)):
            args += [flag, json.dumps(value)]
        else:
            args += [flag, str(value)]
    return args

LOG_PATH = "/var/log/jambu/vllm.log"
PID_PATH = "/var/run/jambu/vllm.pid"

# Pinned, never `latest`: it moved to CUDA 13 under us and landed on a host
# whose driver tops out at 12.2. The main build, not a `-cuNNN` variant: the
# v0.32.0-cu129 variant ships a torchcodec that cannot load (seen live).
DEFAULT_IMAGE = "vllm/vllm-openai:v0.32.0"
VLLM_REPO = "vllm/vllm-openai"

# CUDA behind the bare `vX.Y.Z` / `latest` tags (the image's NVIDIA_REQUIRE_CUDA),
# from the first release that requires it. Older bare tags are left unknown.
_BARE_TAG_CUDA = [((0, 32), 13.0)]
_LATEST_CUDA = 13.0
_VERSION_TAG = re.compile(r"^v?(\d+)\.(\d+)(?:\.\d+)?$")


def vllm_image_cuda(image: str) -> Optional[float]:
    """CUDA a vLLM image needs: the ``-cuNNN`` suffix, else the known bare tags."""
    tag = image_tag(image)
    suffixed = cuda_from_tag(tag)
    if suffixed is not None:
        return suffixed
    if not image_repo(image).endswith(VLLM_REPO):
        return None
    if tag == "latest":
        return _LATEST_CUDA
    match = _VERSION_TAG.match(tag)
    if not match:
        return None
    version = (int(match.group(1)), int(match.group(2)))
    found = None
    for since, cuda in _BARE_TAG_CUDA:
        if version >= since:
            found = cuda
    return found


class VllmRuntime(ModelRuntime):
    engine = "vllm"
    default_image = DEFAULT_IMAGE
    api_key_env = "VLLM_API_KEY"

    def required_cuda(self) -> Optional[float]:
        return vllm_image_cuda(self.image())

    def required_compute_cap(self) -> Optional[float]:
        # vLLM wheels built for CUDA 13+ drop Volta (sm_70) and need Ampere
        # (8.0) or newer; those crash at engine-core init on a V100 or older.
        # CUDA 12.x wheels still run Volta. Pascal (GTX 10xx, 6.1) is below
        # vLLM's floor on any build.
        cuda = self.required_cuda()
        if cuda is None:
            return None
        return 8.0 if cuda >= 13.0 else 7.0

    def runtime_pid_path(self) -> str:
        return PID_PATH

    def runtime_log_path(self) -> str:
        return LOG_PATH

    def env(self) -> dict[str, str]:
        env: dict[str, str] = {
            "HF_HOME": "/workspace/hf",
            "HUGGING_FACE_HUB_CACHE": "/workspace/hf/hub",
            "VLLM_LOGGING_LEVEL": "INFO",
        }
        if self.config.model.trust_remote_code:
            env["VLLM_TRUST_REMOTE_CODE"] = "1"
        return env

    def resolve_tool_call_parser(self) -> "str | None":
        cfg = self.config.runtime
        if cfg.tool_calling == "off":
            return None
        if cfg.tool_call_parser:
            return cfg.tool_call_parser
        return guess_tool_call_parser(self.config.model.id)

    def server_args(self) -> list[str]:
        cfg = self.config
        args: list[str] = [
            "--model",
            cfg.model.id,
            "--host",
            "0.0.0.0",
            "--port",
            str(cfg.runtime.port),
            "--dtype",
            cfg.runtime.dtype,
            "--gpu-memory-utilization",
            str(cfg.runtime.gpu_memory_utilization),
        ]
        if cfg.model.revision:
            args += ["--revision", cfg.model.revision]
        if cfg.model.served_name:
            args += ["--served-model-name", cfg.model.served_name]
        if cfg.runtime.context_length:
            args += ["--max-model-len", str(cfg.runtime.context_length)]
        tp = cfg.runtime.tensor_parallel_size or cfg.compute.gpu.count
        if tp and tp > 1:
            args += ["--tensor-parallel-size", str(tp)]
        if cfg.model.trust_remote_code:
            args.append("--trust-remote-code")
        if cfg.runtime.api_key:
            args += ["--api-key", cfg.runtime.api_key]

        # Agentic flows: /v1/chat/completions and /v1/responses both need the
        # model to actually be able to emit tool calls, which needs a parser
        # matched to the model family.
        parser = self.resolve_tool_call_parser()
        if parser:
            args += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
        if cfg.runtime.tool_server:
            # Lets /v1/responses execute MCP tools server-side instead of the
            # calling agent framework doing it client-side.
            args += ["--tool-server", cfg.runtime.tool_server]

        # Escape hatch for vLLM flags this schema doesn't model explicitly -
        # structured form first, then the raw form, each able to override
        # what came before it (vLLM's argparse takes the last occurrence of
        # a repeated flag).
        args += flatten_vllm_args(cfg.runtime.vllm_args)
        args += list(cfg.runtime.extra_args)
        return args

    def start_command(self) -> str:
        args = " ".join(shlex.quote(a) for a in self.server_args())
        # The interpreter that can import vllm: images that put a Python shim
        # first on PATH (vastai/vllm) keep vllm under /usr/bin/python3.
        find_python = (
            "PY=python3; for p in /usr/bin/python3 python3 python; do "
            "$p -c 'import vllm' 2>/dev/null && PY=$p && break; done"
        )
        return (
            f"mkdir -p $(dirname {LOG_PATH}) && {find_python}; "
            f"nohup $PY -m vllm.entrypoints.openai.api_server {args} "
            f">> {LOG_PATH} 2>&1 & echo $! > {PID_PATH}"
        )

    def activity_probe(self) -> dict:
        # The watchdog runs in the same container, so loopback reaches vLLM
        # without the public port. /metrics is served even with --api-key.
        return {
            "kind": "prometheus",
            "url": f"http://127.0.0.1:{self.config.runtime.port}/metrics",
            "gauges": ["vllm:num_requests_running", "vllm:num_requests_waiting"],
            "counters": ["vllm:request_success_total", "vllm:prompt_tokens_total"],
        }

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        cfg = self.config
        if not (0.1 <= cfg.runtime.gpu_memory_utilization <= 1.0):
            result.error(
                "runtime.gpu_memory_utilization must be between 0.1 and 1.0",
                "runtime.gpu_memory_utilization",
            )
        tp = cfg.runtime.tensor_parallel_size
        if tp and tp > cfg.compute.gpu.count:
            result.error(
                f"runtime.tensor_parallel_size ({tp}) exceeds compute.gpu.count "
                f"({cfg.compute.gpu.count})",
                "runtime.tensor_parallel_size",
            )
        if cfg.runtime.context_length and cfg.runtime.context_length > 131072:
            result.warn(
                "runtime.context_length above 128k needs a lot of KV cache VRAM",
                "runtime.context_length",
            )
        if "/" not in cfg.model.id:
            result.warn(
                f"model.id '{cfg.model.id}' does not look like a HuggingFace repo id",
                "model.id",
            )

        collisions = {
            key.lstrip("-").replace("_", "-") for key in cfg.runtime.vllm_args
        } & _MANAGED_VLLM_FLAGS
        if collisions:
            result.warn(
                f"runtime.vllm_args sets {sorted(collisions)}, which gpu "
                "already manages via other config fields (compute.gpu, model.id, "
                "runtime.tool_calling, ...) - it will still be passed through and "
                "win (last flag wins), but the field meant to control it won't "
                "reflect what's actually running",
                "runtime.vllm_args",
            )

        if cfg.runtime.tool_calling == "on" and not self.resolve_tool_call_parser():
            result.error(
                "runtime.tool_calling is 'on' but no --tool-call-parser could be "
                f"resolved for '{cfg.model.id}'. Set runtime.tool_call_parser "
                "explicitly (see docs/CONFIG.md for the list vLLM supports).",
                "runtime.tool_call_parser",
            )
        elif cfg.runtime.tool_calling == "auto" and not self.resolve_tool_call_parser():
            result.warn(
                f"could not guess a --tool-call-parser for '{cfg.model.id}'; "
                "tool/function calling will be OFF on both /v1/chat/completions "
                "and /v1/responses. Set runtime.tool_call_parser to turn it on.",
                "runtime.tool_call_parser",
            )
        if cfg.runtime.tool_server and not self.resolve_tool_call_parser():
            result.warn(
                "runtime.tool_server is set but no tool-call parser is active; "
                "the model will not be able to invoke it",
                "runtime.tool_server",
            )
        return result

    def describe(self) -> str:
        parser = self.resolve_tool_call_parser()
        tools = f", tools={parser}" if parser else ""
        return f"vLLM :{self.config.runtime.port} ({self.config.model.id}{tools})"
