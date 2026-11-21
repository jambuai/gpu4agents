"""llama.cpp (GGUF) runtime: serves a GGUF chat model with ``llama-server``.

Why this exists: a large class of useful models — abliterated/uncensored
coders, cyber-tuned MoEs — ship only as GGUF, which vLLM cannot load. llama.cpp's
``llama-server`` serves them behind an OpenAI-compatible API
(``/v1/chat/completions`` with tool calling via ``--jinja``), pulls the GGUF
straight from Hugging Face (``-hf <repo>:<quant>``), and exposes the same two
things the lifecycle needs: a ``/health`` endpoint and a Prometheus
``/metrics`` endpoint (``--metrics``) the on-instance watchdog reads to tell
the machine is in use.

Mirrors ``runtime/vllm.py``: the same network-restart launcher packed into the
provider's small onstart field, the same activity-probe shape (different metric
names), and nothing else — gpu only turns on the right ``llama-server`` flags.
"""

from __future__ import annotations

import base64
import shlex
import zlib
from typing import Optional

from ..core.models import ValidationResult
from .base import (
    FETCH_MODEL_TOKEN,
    FETCH_WAIT,
    HOST_NETWORK_FAILURE_PATTERN,
    ModelRuntime,
    THREAD_CAPS,
)
from .vllm import flatten_vllm_args

LOG_PATH = "/var/log/jambu/llama.log"
PID_PATH = "/var/run/jambu/llama.pid"
LAUNCHER_PATH = "/opt/jambu/llama-run.sh"

# Pinned by digest (never `latest`): the official CUDA server build. The tag is
# kept for readability; the digest is what actually resolves. Overridable with
# compute.image. Keep this recent: an older build rejects newer GGUF model
# architectures at load ("unknown model architecture", e.g. qwen35moe). To bump
# it, resolve the live digest of the server-cuda tag and pin that.
DEFAULT_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp:server-cuda"
    "@sha256:22bd49f4cf68236a63c5a187d7d1867ffb08472611ddfbf0392b6842a1b30d44"
)

# llama.cpp CUDA wheels run back to Pascal (6.1); its server images are built
# against CUDA 12.x, which every current datacenter driver provides.
_REQUIRED_CUDA = 12.4

# As many layers as it takes to put the whole model on the GPU. llama.cpp
# clamps an over-large value to the real layer count, so this offloads all of
# them for any model we would rent a GPU for.
_ALL_LAYERS = 999

NETWORK_RESTARTS = 4

LLAMA_ROUTES = ["/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/models"]


class LlamaRuntime(ModelRuntime):
    engine = "llama"
    default_image = DEFAULT_IMAGE
    api_key_env = "LLAMA_API_KEY"

    def required_cuda(self) -> Optional[float]:
        return _REQUIRED_CUDA

    def runtime_pid_path(self) -> str:
        return PID_PATH

    def runtime_log_path(self) -> str:
        return LOG_PATH

    def env(self) -> dict[str, str]:
        # llama.cpp's `-hf` downloader caches under LLAMA_CACHE (not the HF Hub
        # cache dirs) and reads HF_TOKEN for gated repos (injected like vLLM's).
        # Point the cache at the big /workspace volume so a re-rent reuses it.
        return {
            "LLAMA_CACHE": "/workspace/llama-cache",
            "HF_HOME": "/workspace/hf",
            "HUGGING_FACE_HUB_CACHE": "/workspace/hf/hub",
        }

    def _n_gpu_layers(self) -> int:
        n = self.config.runtime.n_gpu_layers
        return _ALL_LAYERS if n is None else n

    def server_args(self) -> list[str]:
        cfg = self.config
        model_ref = cfg.model.id
        if cfg.runtime.quant:
            model_ref = f"{cfg.model.id}:{cfg.runtime.quant}"
        args: list[str] = [
            "-hf",
            model_ref,
            "--host",
            "0.0.0.0",
            "--port",
            str(cfg.runtime.port),
            "--n-gpu-layers",
            str(self._n_gpu_layers()),
            "--metrics",  # Prometheus /metrics for the watchdog activity probe
            "--jinja",  # use the model's own chat template -> tool calling
        ]
        if cfg.runtime.context_length:
            args += ["--ctx-size", str(cfg.runtime.context_length)]
        if cfg.runtime.api_key:
            args += ["--api-key", cfg.runtime.api_key]
        # Escape hatch: structured flags first, then raw argv (last wins).
        args += flatten_vllm_args(cfg.runtime.llama_args)
        args += list(cfg.runtime.extra_args)
        return args

    def fetches_weights(self) -> bool:
        cfg = self.config
        return cfg.runtime.fetch in ("auto", "watchdog") and cfg.lifecycle.watchdog.enabled

    def fetch_request(self) -> Optional[dict]:
        if not self.fetches_weights():
            return None
        cfg = self.config
        return {
            "repo": cfg.model.id,
            "revision": cfg.model.revision or "main",
            "engine": "llama",
            "quant": cfg.runtime.quant,
        }

    def launcher_args(self) -> str:
        """The server's argv, quoted; with a fetch, ``-m <file>`` once it is on disk.

        $D is left unquoted: for a repo with a vision projector it holds
        ``<file> --mmproj <file>`` (the fetcher refuses names with spaces).
        """
        args = self.server_args()
        if not self.fetches_weights():
            return " ".join(shlex.quote(a) for a in args)
        at = args.index("-hf")
        remote = args[at + 1]
        args[at : at + 2] = [FETCH_MODEL_TOKEN]
        return " ".join(shlex.quote(a) for a in args).replace(
            FETCH_MODEL_TOKEN, f"$F ${{D:-{remote}}}"
        )

    def launcher_script(self) -> str:
        """Bash that runs llama-server with the same network-restart loop vLLM
        uses: a dropped connection while it downloads the GGUF restarts it in
        place (the download resumes from disk) instead of losing the machine."""
        args = self.launcher_args()
        fetch = [*FETCH_WAIT, "F=-hf;[ -n \"$D\" ]&&F=-m"] if self.fetches_weights() else []
        return "\n".join(
            [
                *THREAD_CAPS,
                *fetch,
                "L=$1; c=",
                # The image's binary is /app/llama-server and /app is NOT on
                # PATH; resolve it (fall back to the known absolute path).
                "LS=$(command -v llama-server || echo /app/llama-server)",
                "trap '[ -n \"$c\" ] && kill $c; exit 143' TERM INT",
                f"for i in $(seq 0 {NETWORK_RESTARTS}); do",
                "  n=$(wc -l < $L)",
                f'  "$LS" {args} & c=$!',
                "  wait $c; r=$?",
                f"  tail -n +$((n + 1)) $L | tail -40 | grep -qE {shlex.quote(HOST_NETWORK_FAILURE_PATTERN)} || exit $r",
                f"  [ $i -lt {NETWORK_RESTARTS} ] || exit $r",
                '  echo "jambu: the model server hit a network error; restart $((i + 1)) in $((15 * i + 15))s"',
                "  sleep $((15 * i + 15))",
                "done",
                "",
            ]
        )

    def start_command(self) -> str:
        # Deflated: the boot script rides in a size-capped field (Vast: 16 KB).
        blob = base64.b85encode(zlib.compress(self.launcher_script().encode(), 9)).decode()
        unpack = "import base64,sys,zlib;open(sys.argv[1],'wb').write(zlib.decompress(base64.b85decode(sys.argv[2])))"
        return (
            # Their directories come from the boot script's own mkdir.
            f"{self._fetch_prefix()}PY=$(command -v python3 || command -v python); "
            f"\"$PY\" -c \"{unpack}\" {LAUNCHER_PATH} '{blob}'; touch {LOG_PATH}; "
            f'nohup bash {LAUNCHER_PATH} {LOG_PATH} >> {LOG_PATH} 2>&1 & echo $! > {PID_PATH}'
        )

    def routes(self) -> list[str]:
        return list(LLAMA_ROUTES)

    def activity_probe(self) -> dict:
        # The watchdog shares the container, so loopback reaches llama-server's
        # /metrics (served once --metrics is set). A request in flight shows as
        # a non-zero gauge; a finished one grows the token counters.
        return {
            "kind": "prometheus",
            "url": f"http://127.0.0.1:{self.config.runtime.port}/metrics",
            "gauges": ["llamacpp:requests_processing", "llamacpp:requests_deferred"],
            "counters": ["llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total"],
        }

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        cfg = self.config
        if not cfg.runtime.quant:
            result.error(
                "runtime.quant is required for the llama engine: the GGUF quant "
                "tag to pull from model.id (e.g. 'Q5_K_XL' -> -hf <id>:Q5_K_XL)",
                "runtime.quant",
            )
        if "/" not in cfg.model.id:
            result.warn(
                f"model.id '{cfg.model.id}' does not look like a HuggingFace repo id",
                "model.id",
            )
        elif "gguf" not in cfg.model.id.lower():
            result.warn(
                f"model.id '{cfg.model.id}' has no 'GGUF' marker; the llama engine "
                "needs a GGUF repo, not a safetensors one",
                "model.id",
            )
        return result

    def describe(self) -> str:
        quant = f":{self.config.runtime.quant}" if self.config.runtime.quant else ""
        return f"llama-server :{self.config.runtime.port} ({self.config.model.id}{quant})"
