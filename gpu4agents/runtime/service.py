"""Model servers that run as service jobs (``runtime.engine: hf | custom``).

The boot script starts nothing for these engines. Once the machine is up, the
engine uploads the server's directory to the job runner on the watchdog and
starts it as a *service*: a job that is not a workload (it does not hold the
machine up by running) and that becomes the runtime the watchdog watches.

- The machine stays up while clients send requests: the server's /metrics
  counts them and the watchdog's activity probe reads it over loopback.
- Its process is the runtime's pid, so a server that dies while loading ends
  the startup wait with its log, as a crashed vLLM does.
- What runs is identified by a digest of the code, the command and the
  settings. `gpu setup` on a machine already running another digest stops
  the old server and starts the new one: changing the model or the code needs
  no new machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.errors import ConfigError
from ..core.job import bundle_files, load_manifest
from ..core.models import ValidationResult
from ..serve import HF_SERVER_DIR, ROUTES
from .base import NoRuntime

SERVICE_LABEL = "service"
PID_PATH = "/var/run/jambu/service.pid"
LOG_PATH = "/var/log/jambu/service.log"
# The bundled server's package marker is for importing it here, not for the machine.
HF_BUNDLE_EXCLUDE = ("__init__.py",)


@dataclass
class ServiceSpec:
    """What to upload and how to start it."""

    directory: Path
    command: str
    setup: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=list)

    def digest(self) -> str:
        """Identity of what would run: the code, the command, the settings."""
        sha = hashlib.sha256()
        sha.update(json.dumps([self.command, self.setup, sorted(self.env.items())]).encode())
        for path, relative in bundle_files(self.directory, self.exclude):
            sha.update(relative.encode() + b"\0")
            sha.update(hashlib.sha256(path.read_bytes()).digest())
        return sha.hexdigest()[:16]


class ServiceRuntime(NoRuntime):
    """Shared by the engines whose server is deployed after boot."""

    service = True
    api_key_env = "JAMBU_API_KEY"

    def ports(self) -> list[int]:
        return [self.config.runtime.port]

    def start_command(self) -> str:
        return "true"

    def health_url(self, base_url: str) -> str:
        return base_url.rstrip("/") + self.health_path()

    def runtime_pid_path(self) -> str:
        return PID_PATH

    def runtime_log_path(self) -> str:
        return LOG_PATH

    def activity_probe(self) -> dict:
        return {
            "kind": "prometheus",
            "url": f"http://127.0.0.1:{self.config.runtime.port}/metrics",
            "gauges": ["jambu_requests_running"],
            "counters": ["jambu_requests_total"],
        }

    def base_env(self) -> dict[str, str]:
        cfg = self.config
        env = {"PORT": str(cfg.runtime.port), "HF_HOME": "/workspace/hf"}
        if cfg.model.id:
            env["JAMBU_MODEL_ID"] = cfg.model.id
            env["JAMBU_SERVED_NAME"] = cfg.model.served_name or cfg.model.id
        return env

    def service_spec(self) -> ServiceSpec:  # pragma: no cover - abstract
        raise NotImplementedError

    def routes(self) -> list[str]:
        return []


class HfRuntime(ServiceRuntime):
    """gpu's own OpenAI-compatible server for any Hugging Face model."""

    engine = "hf"

    def service_spec(self) -> ServiceSpec:
        cfg = self.config
        env = self.base_env()
        env.update(
            {
                "JAMBU_TASK": cfg.model.task,
                "JAMBU_DTYPE": cfg.runtime.dtype,
                "JAMBU_PACKAGES": " ".join(cfg.runtime.packages),
            }
        )
        if cfg.model.revision:
            env["JAMBU_REVISION"] = cfg.model.revision
        if cfg.model.trust_remote_code:
            env["JAMBU_TRUST_REMOTE_CODE"] = "1"
        return ServiceSpec(
            directory=HF_SERVER_DIR,
            command='exec "$(command -v python3 || command -v python)" server.py',
            setup="bash install.sh",
            env=env,
            exclude=list(HF_BUNDLE_EXCLUDE),
        )

    def routes(self) -> list[str]:
        task = self.config.model.task
        if task == "auto":
            return []
        return list(ROUTES[task])

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        cfg = self.config
        if "/" not in cfg.model.id:
            result.warn(
                f"model.id '{cfg.model.id}' does not look like a HuggingFace repo id", "model.id"
            )
        if cfg.model.task == "auto":
            result.warn(
                "model.task is auto: the server reads the model card's pipeline_tag at start. "
                "Set it (image, speech, transcription, embeddings, chat, pipeline) to know "
                "the routes before renting",
                "model.task",
            )
        return result

    def describe(self) -> str:
        task = self.config.model.task
        return f"hf server :{self.config.runtime.port} ({self.config.model.id}, {task})"


class CustomRuntime(ServiceRuntime):
    """A server the agent wrote, in runtime.service_dir (SERVING.md)."""

    engine = "custom"
    DEFAULT_DIR = "server"

    def directory(self) -> Path:
        raw = self.config.runtime.service_dir or self.DEFAULT_DIR
        path = Path(raw).expanduser()
        return path if path.is_absolute() else self.config.project_dir / path

    def service_spec(self) -> ServiceSpec:
        directory = self.directory()
        if not directory.is_dir():
            raise ConfigError(
                f"runtime.service_dir {directory} does not exist; `gpu init --engine custom` "
                "writes a starting server there"
            )
        manifest = load_manifest(directory)
        if not manifest.run:
            raise ConfigError(f"{directory}/gpujob.yaml has no run: (the command that starts the server)")
        env = self.base_env()
        env.update(manifest.env)
        return ServiceSpec(
            directory=directory,
            command=manifest.run,
            setup=manifest.setup,
            env=env,
            exclude=list(manifest.exclude),
        )

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        try:
            self.service_spec()
        except ConfigError as exc:
            result.error(str(exc), "runtime.service_dir")
        return result

    def describe(self) -> str:
        return f"custom server :{self.config.runtime.port} ({self.directory().name}/)"
