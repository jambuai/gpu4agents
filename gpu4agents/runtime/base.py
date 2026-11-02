"""Runtime contract (spec section 19).

Provider answers "where does the compute run?".
Runtime answers "how does the model run?".
Model answers "what model runs?".
These stay independent; a runtime never talks to a provider API.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from ..core.config import RuntimeConfig
from ..core.errors import ConfigError
from ..core.models import ValidationResult

ENDPOINT_NAME = "model"

# `-cu129` -> 12.9, `cu130` -> 13.0, `-cuda12.1-` -> 12.1: the CUDA toolkit an
# image tag was built for.
_CU_TAG = re.compile(r"(?:^|[-_.])cu(\d{2})(\d)(?![0-9])")
# `cuda12.1` (pytorch) and `cuda-13.0` (vastai/vllm) spellings.
_CUDA_TAG = re.compile(r"cuda[-_]?(\d+)\.(\d+)")


def image_tag(image: str) -> str:
    """The tag of ``repo[:tag][@digest]``; an untagged reference means ``latest``."""
    name = image.split("@", 1)[0]
    last = name.rsplit("/", 1)[-1]
    return last.split(":", 1)[1] if ":" in last else "latest"


def image_repo(image: str) -> str:
    name = image.split("@", 1)[0]
    head, _, last = name.rpartition("/")
    last = last.split(":", 1)[0]
    return f"{head}/{last}" if head else last


def cuda_from_tag(tag: str) -> Optional[float]:
    match = _CU_TAG.search(tag) or _CUDA_TAG.search(tag)
    if not match:
        return None
    return float(f"{int(match.group(1))}.{match.group(2)}")


def parse_cuda(value: Any) -> Optional[float]:
    """``"12.4"`` / ``12.4`` / ``None`` -> float or None (garbage is None)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def max_cuda(*values: Any) -> Optional[float]:
    known = [v for v in (parse_cuda(x) for x in values) if v is not None]
    return max(known) if known else None


class ModelRuntime(ABC):
    """Describes how to boot a model server inside the provisioned container."""

    engine: str = "abstract"
    default_image: str = ""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        # Set by the engine from the provider's preferred_images.
        self.provider_image: Optional[str] = None

    # -- container shape ----------------------------------------------------

    def image(self) -> str:
        return self.config.compute.image or self.provider_image or self.default_image

    def ports(self) -> list[int]:
        return [self.config.runtime.port]

    def env(self) -> dict[str, str]:
        return {}

    # The environment variable through which the server takes its API key;
    # None for a runtime that cannot require one.
    api_key_env: Optional[str] = None

    @abstractmethod
    def start_command(self) -> str:
        """Shell command that launches the model server in the background."""

    # -- health -------------------------------------------------------------

    def health_path(self) -> str:
        return self.config.health.path

    def health_url(self, base_url: str) -> str:
        return base_url.rstrip("/") + self.health_path()

    def base_url(self, endpoint_url: str) -> str:
        return endpoint_url.rstrip("/")

    def activity_probe(self) -> Optional[dict]:
        """How the on-instance watchdog can tell the runtime is serving requests.

        None means the runtime exposes no such signal; only workloads, startup
        and locks then count as activity.
        """
        return None

    def required_cuda(self) -> Optional[float]:
        """Minimum host CUDA (driver) version the image needs, None if unknown.

        The offer filter is raised to this, so an image built for a newer CUDA
        is never placed on a host whose driver cannot run it.
        """
        return cuda_from_tag(image_tag(self.image()))

    def required_compute_cap(self) -> Optional[float]:
        """Minimum GPU compute capability (architecture) the runtime needs.

        None if unknown. The offer filter is raised to this, so an image whose
        kernels are built for newer architectures is never placed on a GPU too
        old to run them (which the host driver's CUDA alone does not catch: a
        Volta card behind a CUDA-13 driver still fails at engine startup).
        """
        return None

    # -- on-instance liveness -------------------------------------------------

    def runtime_pid_path(self) -> Optional[str]:
        """File holding the model server's pid, for the watchdog's liveness check."""
        return None

    def runtime_log_path(self) -> Optional[str]:
        """The model server's log on the instance, served by the watchdog's /logs."""
        return None

    def validate(self) -> ValidationResult:
        return ValidationResult()

    def describe(self) -> str:
        return f"{self.engine} :{self.config.runtime.port}"


class NoRuntime(ModelRuntime):
    """`runtime.engine: none` - provision bare compute, start nothing."""

    engine = "none"
    default_image = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"

    def ports(self) -> list[int]:
        return []

    def start_command(self) -> str:
        return "true"

    def health_url(self, base_url: str) -> str:  # pragma: no cover - unused
        return base_url


class RuntimeRegistry:
    def __init__(self) -> None:
        self._runtimes: dict[str, Type[ModelRuntime]] = {"none": NoRuntime}

    def register(self, runtime_cls: Type[ModelRuntime]) -> Type[ModelRuntime]:
        self._runtimes[runtime_cls.engine] = runtime_cls
        return runtime_cls

    def names(self) -> list[str]:
        return sorted(self._runtimes)

    def get(self, config: RuntimeConfig, engine: Optional[str] = None) -> ModelRuntime:
        key = engine or config.runtime.engine
        try:
            return self._runtimes[key](config)
        except KeyError:
            known = ", ".join(self.names())
            raise ConfigError(f"unknown runtime engine '{key}' (available: {known})") from None


runtime_registry = RuntimeRegistry()
