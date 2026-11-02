"""jambu.yaml schema and loader (spec section 4).

jambu.yaml is the declarative desired-state Source of Truth. It describes
*what* should run and *how long it may live*. It never contains credentials
and never contains provider-specific API details.

jambu.yaml is a SHARED manifest across Jambu.ai Lab sub-projects, not a file
this tool owns exclusively - our config lives under one namespaced top-level
key (``gpu_runtime:``), so other Jambu tools can add their own keys to the
same file without colliding with ours:

    # jambu.yaml
    gpu_runtime:
      version: 1
      model: {id: ...}
      ...
    some_other_jambu_tool:
      ...

The legacy, unnamespaced ``config.yml`` / ``config.yaml`` (the whole document
IS the gpu-runtime config, no wrapper key) is still read for projects that
predate this convention or that intentionally want a gpu-runtime-only file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .durations import parse_duration
from .errors import ConfigError
from .paths import project_state_dir

# jambu.yaml (namespaced, shared with other Jambu Lab tools) is searched
# first; config.yml (flat, gpu-runtime-only) is the legacy fallback.
CONFIG_FILENAMES = ("jambu.yaml", "jambu.yml", "config.yml", "config.yaml")
GPU_RUNTIME_KEY = "gpu_runtime"

# The model catalog (spec: "Choosing a model" doc) - a project's jambu.yaml
# references one entry by name instead of redeclaring the whole model/compute/
# runtime block per model, per project. Searched the same way as jambu.yaml
# itself (this directory, then parents), so one catalog can serve every
# sub-project in a repo without duplication.
CATALOG_FILENAMES = ("jambu.models.yaml", "jambu.models.yml")
PROFILE_KEY = "model_profile"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Duration(str):
    """A duration string; ``.seconds`` is resolved at validation time."""


def _duration(field_name: str):
    def _validate(cls, v):  # noqa: N805
        try:
            parse_duration(v)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return v

    return field_validator(field_name)(classmethod(_validate))


class ProviderConfig(_Base):
    name: str = "vast"
    profile: str = "default"
    options: dict[str, Any] = Field(default_factory=dict)


class GpuConfig(_Base):
    min_vram_gb: int = 24
    count: int = 1
    name_filter: Optional[str] = None
    cuda_min: Optional[str] = None
    # Minimum GPU compute capability (architecture), e.g. 8.0 to require
    # Ampere or newer. Raised to what the runtime needs; see
    # ModelRuntime.required_compute_cap. Leave unset to accept the runtime's
    # own floor.
    compute_cap_min: Optional[float] = None

    @field_validator("count")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("compute.gpu.count must be >= 1")
        return v


class ComputeConfig(_Base):
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    disk_gb: int = 80
    region: Optional[str] = None
    spot: bool = False
    image: Optional[str] = None
    # Machines to try when one fails to boot (bad network, stuck image pull)
    # before setup gives up. Each attempt rents a different host.
    host_attempts: int = 3

    @field_validator("host_attempts")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("compute.host_attempts must be >= 1")
        return v


class ModelConfig(_Base):
    id: str
    revision: Optional[str] = None
    hf_token_env: Optional[str] = "HF_TOKEN"
    trust_remote_code: bool = False
    served_name: Optional[str] = None


class RuntimeConfigSection(_Base):
    engine: Literal["vllm", "none"] = "vllm"
    port: int = 8000
    context_length: Optional[int] = 16384
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: Optional[int] = None
    api_key: Optional[str] = None

    # Escape hatch for any vLLM CLI flag this schema doesn't model explicitly
    # (vLLM adds flags faster than any fixed contract can track). Two forms,
    # composable, applied in this order - later wins on conflict:
    #   vllm_args: a structured {flag: value} map, flattened into --flag
    #     value pairs (see runtime/vllm.py::flatten_vllm_args for the exact
    #     rules - booleans, dicts/lists as JSON, dashes vs underscores).
    #   extra_args: a raw argv list, appended verbatim - for anything the
    #     structured form can't express (repeated flags, exotic syntax).
    vllm_args: dict[str, Any] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)

    # Agentic flows ride on vLLM's own OpenAI-compatible server - both
    # /v1/chat/completions (with tools) and /v1/responses (the OpenAI
    # Responses API) are already implemented there. Nothing for gpu to
    # build; these fields only turn on the right vLLM flags.
    #   auto -> guess a --tool-call-parser from model.id, enable if recognized
    #   on   -> require a resolvable parser (explicit tool_call_parser or a
    #           recognized model.id); fail validation otherwise
    #   off  -> never pass tool-calling flags
    tool_calling: Literal["auto", "on", "off"] = "auto"

    @field_validator("tool_calling", mode="before")
    @classmethod
    def _tool_calling_yaml_bool(cls, v: Any) -> Any:
        # YAML 1.1 (what PyYAML's safe_load speaks) reads a bare `on`/`off`
        # as the boolean True/False, not the string - a classic footgun (the
        # "Norway problem"). Accept the boolean so an unquoted `on`/`off` in
        # jambu.yaml still does what it looks like it does, instead of a
        # confusing "Input should be 'auto', 'on' or 'off'" error.
        if v is True:
            return "on"
        if v is False:
            return "off"
        return v

    tool_call_parser: Optional[str] = None
    # MCP server vLLM should route Responses-API tool calls to server-side
    # (a URL, or "demo" for vLLM's bundled demo server). Leave unset to let
    # the calling agent frameworks execute their own tools client-side.
    tool_server: Optional[str] = None


class LockConfig(_Base):
    enabled: bool = False
    default_ttl: Optional[str] = "2h"

    _v_ttl = _duration("default_ttl")


class WatchdogConfig(_Base):
    """Independent lifecycle enforcement on the instance (spec section 14)."""

    enabled: bool = True
    port: int = 8777
    interval: str = "60s"
    # Env var holding the credential the watchdog uses to stop/destroy itself.
    credential_env: Optional[str] = None

    _v_interval = _duration("interval")


class LifecycleConfig(_Base):
    max_lifetime: Optional[str] = "6h"
    idle_timeout: Optional[str] = "15m"
    auto_stop: bool = True
    auto_destroy: bool = False
    cleanup_on_setup_failure: bool = True
    allow_indefinite_lock: bool = False
    heartbeat_interval: str = "30s"
    heartbeat_grace: str = "5m"
    ssh_counts_as_activity: bool = False
    # Requests served by the model runtime count as activity, so an endpoint
    # in active use by something other than `gpu run` is not idled out.
    traffic_counts_as_activity: bool = True
    # How long an instance may sit STOPPED before it is destroyed. Matters
    # where a stopped instance still bills (storage); null keeps it forever.
    stopped_ttl: Optional[str] = "24h"
    lock: LockConfig = Field(default_factory=LockConfig)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)

    _v_max = _duration("max_lifetime")
    _v_idle = _duration("idle_timeout")
    _v_hb = _duration("heartbeat_interval")
    _v_grace = _duration("heartbeat_grace")
    _v_stopped_ttl = _duration("stopped_ttl")

    @model_validator(mode="after")
    def _sane(self) -> "LifecycleConfig":
        idle = parse_duration(self.idle_timeout)
        life = parse_duration(self.max_lifetime)
        if idle is not None and life is not None and idle > life:
            raise ValueError("lifecycle.idle_timeout must not exceed lifecycle.max_lifetime")
        if not self.auto_stop and not self.auto_destroy:
            # Not fatal, but the guard becomes advisory only.
            pass
        return self


class HealthConfig(_Base):
    interval: str = "60s"
    # How long the machine may take to reach running (mostly the image pull).
    boot_timeout: str = "30m"
    # How long that pull may go without progress before the host counts as
    # stalled; a stalled pull never resumes, so waiting out boot_timeout
    # only bills the dead host longer.
    pull_stall_timeout: str = "10m"
    # How long the model server may then take to answer its health check.
    startup_timeout: str = "10m"
    path: str = "/health"

    _v_interval = _duration("interval")
    _v_boot = _duration("boot_timeout")
    _v_stall = _duration("pull_stall_timeout")
    _v_startup = _duration("startup_timeout")


class BudgetConfig(_Base):
    max_hourly_cost_usd: Optional[float] = None
    max_session_cost_usd: Optional[float] = None


class WorkloadConfig(_Base):
    mode: Literal["local", "remote"] = "local"
    env: dict[str, str] = Field(default_factory=dict)
    workdir: Optional[str] = None
    sync: bool = False


class RuntimeConfig(_Base):
    """The whole parsed gpu-runtime config (jambu.yaml's gpu_runtime: section, or a legacy config.yml)."""

    version: int = 1
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    model: ModelConfig
    runtime: RuntimeConfigSection = Field(default_factory=RuntimeConfigSection)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)

    # Populated by the loader; not part of the YAML document.
    source_path: Optional[Path] = Field(default=None, exclude=True)
    profile_name: Optional[str] = Field(default=None, exclude=True)
    catalog_path: Optional[Path] = Field(default=None, exclude=True)

    @field_validator("version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"unsupported config version: {v} (expected 1)")
        return v

    # -- convenience accessors, all in seconds ------------------------------

    @property
    def idle_timeout_s(self) -> Optional[float]:
        return parse_duration(self.lifecycle.idle_timeout)

    @property
    def max_lifetime_s(self) -> Optional[float]:
        return parse_duration(self.lifecycle.max_lifetime)

    @property
    def stopped_ttl_s(self) -> Optional[float]:
        return parse_duration(self.lifecycle.stopped_ttl)

    @property
    def heartbeat_interval_s(self) -> float:
        return parse_duration(self.lifecycle.heartbeat_interval) or 30.0

    @property
    def heartbeat_grace_s(self) -> float:
        return parse_duration(self.lifecycle.heartbeat_grace) or 300.0

    @property
    def watchdog_interval_s(self) -> float:
        return parse_duration(self.lifecycle.watchdog.interval) or 60.0

    @property
    def health_interval_s(self) -> float:
        return parse_duration(self.health.interval) or 15.0

    @property
    def boot_timeout_s(self) -> float:
        return parse_duration(self.health.boot_timeout) or 1800.0

    @property
    def pull_stall_timeout_s(self) -> Optional[float]:
        return parse_duration(self.health.pull_stall_timeout)

    @property
    def startup_timeout_s(self) -> float:
        return parse_duration(self.health.startup_timeout) or 600.0

    @property
    def lock_default_ttl_s(self) -> Optional[float]:
        return parse_duration(self.lifecycle.lock.default_ttl)

    @property
    def project_dir(self) -> Path:
        return self.source_path.parent if self.source_path else Path.cwd()

    @property
    def state_dir(self) -> Path:
        return project_state_dir(self.project_dir)

    def fingerprint(self) -> str:
        """Identity of the *desired* runtime.

        Two configs with the same fingerprint may reuse the same instance
        (spec section 23 idempotency); a change means the running instance is
        incompatible with the requested runtime.

        MUST cover every field that feeds the actual `vllm serve` command
        line (see runtime/vllm.py::server_args()) - not just model/compute.
        The onstart script (and the command it launches) is baked in at
        instance CREATION time and re-executed verbatim on every subsequent
        provider-level start, INCLUDING a plain restart of a stopped
        instance - it is never regenerated for an existing instance. A field
        missing here means changing it (e.g. runtime.tool_calling) silently
        keeps serving the OLD command forever, no matter how many times the
        instance is stopped and restarted, while `gpu status` reports
        the NEW config as if it were live. Confirmed the hard way: flipping
        tool_calling on for an already-provisioned instance did nothing until
        this was fixed - only `--recreate` regenerates the onstart script.
        """
        import hashlib
        import json

        material = {
            "provider": self.provider.name,
            "gpu": self.compute.gpu.model_dump(),
            "disk_gb": self.compute.disk_gb,
            "image": self.compute.image,
            "model": {
                "id": self.model.id,
                "revision": self.model.revision,
                "trust_remote_code": self.model.trust_remote_code,
                "served_name": self.model.served_name,
            },
            "runtime": {
                "engine": self.runtime.engine,
                "port": self.runtime.port,
                "context_length": self.runtime.context_length,
                "dtype": self.runtime.dtype,
                "gpu_memory_utilization": self.runtime.gpu_memory_utilization,
                "tensor_parallel_size": self.runtime.tensor_parallel_size,
                "api_key": self.runtime.api_key,
                "tool_calling": self.runtime.tool_calling,
                "tool_call_parser": self.runtime.tool_call_parser,
                "tool_server": self.runtime.tool_server,
                "vllm_args": self.runtime.vllm_args,
                "extra_args": self.runtime.extra_args,
            },
        }
        blob = json.dumps(material, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def find_config(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from ``start`` looking for a config file."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def find_catalog(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for a model catalog file."""
    current = start.resolve()
    for directory in [current, *current.parents]:
        for name in CATALOG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_catalog(path: Path) -> dict[str, Any]:
    """Load a model catalog: ``{profile-name: config-fragment}``."""
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path}: expected a YAML mapping of profile-name -> config fragment"
        )
    return _expand_env(raw)


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` onto ``base``; override wins on conflict.

    Dicts merge key by key (recursively); anything else (including lists) is
    replaced wholesale by ``override`` when present - a profile's
    ``extra_args`` list is not concatenated with the project's, it's swapped.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value) if key in merged else value
        return merged
    return override


def _resolve_profile(body: dict[str, Any], config_dir: Path) -> tuple[dict[str, Any], Optional[str], Optional[Path]]:
    """Merge a ``model_profile:`` reference (if any) into ``body``.

    The project's own fields always win over the profile's - a profile is a
    set of defaults, not a lock. Returns (merged_body, profile_name, catalog_path).
    """
    profile_name = body.get(PROFILE_KEY)
    if profile_name is None:
        return body, None, None
    if not isinstance(profile_name, str):
        raise ConfigError(f"'{PROFILE_KEY}' must be a string, got {type(profile_name).__name__}")

    catalog_path = find_catalog(config_dir)
    if catalog_path is None:
        raise ConfigError(
            f"{PROFILE_KEY}: '{profile_name}' is set, but no "
            f"{'/'.join(CATALOG_FILENAMES)} was found in {config_dir} or any parent. "
            "Create one (see docs/CHOOSING_A_MODEL.md) or remove model_profile "
            "and declare model:/compute: directly."
        )
    catalog = load_catalog(catalog_path)
    if profile_name not in catalog:
        available = ", ".join(sorted(catalog)) or "<none defined>"
        raise ConfigError(
            f"{PROFILE_KEY} '{profile_name}' not found in {catalog_path} "
            f"(available: {available})"
        )
    fragment = catalog[profile_name]
    if not isinstance(fragment, dict):
        raise ConfigError(f"{catalog_path}: profile '{profile_name}' must be a mapping")

    overrides = {k: v for k, v in body.items() if k != PROFILE_KEY}
    merged = deep_merge(fragment, overrides)
    return merged, profile_name, catalog_path


def load_config(path: Optional[Path] = None, start: Optional[Path] = None) -> RuntimeConfig:
    """Load and validate jambu.yaml (or legacy config.yml), raising ConfigError
    with a readable message.
    """
    resolved = Path(path).resolve() if path else find_config(start)
    if resolved is None:
        raise ConfigError(
            "no jambu.yaml (or config.yml) found in this directory or any "
            "parent. Run `gpu init` to create one."
        )
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {resolved}")

    try:
        raw = yaml.safe_load(resolved.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved}: expected a YAML mapping at the top level")

    raw = _expand_env(raw)

    # jambu.yaml is shared with other Jambu Lab tools: our section lives
    # under gpu_runtime: and everything else in the document belongs to
    # someone else. A bare config.yml (no such key) IS the whole document.
    if GPU_RUNTIME_KEY in raw:
        body = raw[GPU_RUNTIME_KEY]
        if not isinstance(body, dict):
            raise ConfigError(
                f"{resolved}: '{GPU_RUNTIME_KEY}:' must be a mapping, got "
                f"{type(body).__name__}"
            )
    else:
        body = raw

    body, profile_name, catalog_path = _resolve_profile(body, resolved.parent)

    try:
        config = RuntimeConfig.model_validate(body)
    except Exception as exc:  # pydantic.ValidationError
        raise ConfigError(f"{resolved}: {_format_validation_error(exc)}") from exc

    config.source_path = resolved
    config.profile_name = profile_name
    config.catalog_path = catalog_path
    return config


def _expand_env(node: Any) -> Any:
    """Expand ``${VAR}`` references inside string values (never credentials in the file)."""
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str) and "${" in node:
        return os.path.expandvars(node)
    return node


def _format_validation_error(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    lines = []
    for err in errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        lines.append(f"{loc}: {err.get('msg')}")
    return "invalid configuration\n  - " + "\n  - ".join(lines)


DEFAULT_CONFIG_TEMPLATE = """\
# jambu.yaml - shared across Jambu.ai Lab sub-projects. This tool only reads
# the gpu_runtime: key below; add other tools' own top-level keys alongside
# it without conflict.
gpu_runtime:
  version: 1

  provider:
    name: vast
    profile: default

  compute:
    gpu:
      min_vram_gb: 24
      count: 1
    disk_gb: 80

  model:
    id: {model_id}

  runtime:
    engine: vllm
    port: 8000
    context_length: 16384

  lifecycle:
    max_lifetime: 6h
    idle_timeout: 15m
    stopped_ttl: 24h
    auto_stop: true
    auto_destroy: false
    traffic_counts_as_activity: true
    cleanup_on_setup_failure: true
    allow_indefinite_lock: false
    lock:
      enabled: false
    watchdog:
      enabled: true
      port: 8777
      interval: 60s

  health:
    interval: 60s
    boot_timeout: 30m
    pull_stall_timeout: 10m
    startup_timeout: 10m

  budget:
    max_hourly_cost_usd: 1.00
    max_session_cost_usd: 5.00
"""

# Legacy shape: the whole document IS the gpu-runtime config, no wrapper key.
# Still accepted by load_config() for config.yml / config.yaml.
DEFAULT_CONFIG_TEMPLATE_FLAT = """\
version: 1

provider:
  name: vast
  profile: default

compute:
  gpu:
    min_vram_gb: 24
    count: 1
  disk_gb: 80

model:
  id: {model_id}

runtime:
  engine: vllm
  port: 8000
  context_length: 16384

lifecycle:
  max_lifetime: 6h
  idle_timeout: 15m
  stopped_ttl: 24h
  auto_stop: true
  auto_destroy: false
  traffic_counts_as_activity: true
  cleanup_on_setup_failure: true
  allow_indefinite_lock: false
  lock:
    enabled: false
  watchdog:
    enabled: true
    port: 8777
    interval: 60s

health:
  interval: 60s
  boot_timeout: 30m
  pull_stall_timeout: 10m
  startup_timeout: 10m

budget:
  max_hourly_cost_usd: 1.00
  max_session_cost_usd: 5.00
"""
