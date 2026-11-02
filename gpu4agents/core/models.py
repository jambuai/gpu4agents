"""Provider-independent domain models (spec sections 6-8).

Nothing here may reference a concrete provider. Adapters translate their own
payloads into these shapes and back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional

from .policy import (
    BUSY,
    DESTROYED,
    FAILED,
    IDLE,
    PROVISIONING,
    READY,
    STARTING,
    STOPPED,
    UNKNOWN,
)


# Every instance this tool creates carries a label with this prefix. It is how
# account-wide commands tell our instances from everything else in the account.
JAMBU_LABEL_PREFIX = "jambu-"


def is_jambu_label(label: Optional[str]) -> bool:
    return bool(label) and str(label).startswith(JAMBU_LABEL_PREFIX)


class InstanceState(str, Enum):
    PROVISIONING = PROVISIONING
    STARTING = STARTING
    READY = READY
    BUSY = BUSY
    IDLE = IDLE
    STOPPED = STOPPED
    FAILED = FAILED
    DESTROYED = DESTROYED
    UNKNOWN = UNKNOWN

    @property
    def is_terminal(self) -> bool:
        return self in (InstanceState.DESTROYED,)

    @property
    def is_billable_compute(self) -> bool:
        return self in (
            InstanceState.PROVISIONING,
            InstanceState.STARTING,
            InstanceState.READY,
            InstanceState.BUSY,
            InstanceState.IDLE,
        )


@dataclass
class ProviderCapabilities:
    """What an adapter can actually do (spec section 7).

    The core branches on these flags, never on provider names.
    """

    stop: bool = False
    destroy: bool = True
    persistent_disk: bool = False
    spot_instances: bool = False
    ssh: bool = False
    public_ports: bool = False
    gpu_selection: bool = False
    pricing_query: bool = False
    remote_exec: bool = False
    logs: bool = False
    # False when a stopped instance still costs money (spec section 25).
    stop_eliminates_billing: bool = True
    # Can enumerate this tool's instances account-wide (`gpu ls`, `gpu sweep`).
    list_instances: bool = False

    def as_row(self) -> dict[str, str]:
        return {k: ("yes" if v else "no") for k, v in self.__dict__.items()}


@dataclass
class GpuSpec:
    min_vram_gb: int = 24
    count: int = 1
    name_filter: Optional[str] = None
    cuda_min: Optional[str] = None
    # Minimum GPU compute capability (architecture), e.g. 8.0 for Ampere. The
    # host driver's CUDA (``cuda_min``) is not the same thing: a Pascal/Volta
    # card can sit behind a CUDA-13 driver yet still be too old for a runtime
    # image built for newer architectures, which then dies at engine startup.
    compute_cap_min: Optional[float] = None


@dataclass
class ComputeSpec:
    """The normalized "give me a machine like this" request."""

    gpu: GpuSpec
    disk_gb: int = 80
    image: str = ""
    ports: list[int] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    onstart: str = ""
    label: str = "jambu-gpu"
    spot: bool = False
    region: Optional[str] = None
    max_hourly_cost_usd: Optional[float] = None
    # Machines (Offer.host_id) not to rent: hosts that already failed to boot.
    exclude_hosts: list[str] = field(default_factory=list)
    # Free-form provider hints; adapters ignore keys they do not understand.
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Offer:
    """A normalized purchasable capacity offer."""

    id: str
    gpu_name: str
    gpu_count: int
    gpu_vram_gb: float
    disk_gb: float
    hourly_cost_usd: float
    region: Optional[str] = None
    score: Optional[float] = None
    # Highest CUDA version the host's driver supports, when the provider says.
    cuda_max: Optional[float] = None
    # The GPU's compute capability (architecture), e.g. 8.6 for an RTX 3090,
    # when the provider says. Distinct from cuda_max (the driver's CUDA).
    compute_cap: Optional[float] = None
    # The physical machine behind the offer, when the provider says. Offers
    # on one machine share its network, disk and Docker daemon.
    host_id: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Endpoint:
    """A reachable network address on the instance."""

    name: str
    url: str
    internal_port: int
    external_port: int


@dataclass
class Instance:
    """Normalized handle on remote compute. Provider ids never leak semantics."""

    id: str
    provider: str
    state: InstanceState = InstanceState.PROVISIONING
    created_at: float = field(default_factory=time.time)
    gpu_name: str = ""
    gpu_count: int = 0
    hourly_cost_usd: Optional[float] = None
    host: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_user: str = "root"
    endpoints: dict[str, Endpoint] = field(default_factory=dict)
    label: str = ""
    # The machine it runs on (see Offer.host_id).
    host_id: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    def endpoint_url(self, name: str) -> Optional[str]:
        ep = self.endpoints.get(name)
        return ep.url if ep else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "state": self.state.value,
            "created_at": self.created_at,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "hourly_cost_usd": self.hourly_cost_usd,
            "host": self.host,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "ssh_user": self.ssh_user,
            "label": self.label,
            "host_id": self.host_id,
            "endpoints": {
                name: {
                    "url": ep.url,
                    "internal_port": ep.internal_port,
                    "external_port": ep.external_port,
                }
                for name, ep in self.endpoints.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        endpoints = {
            name: Endpoint(
                name=name,
                url=ep["url"],
                internal_port=int(ep.get("internal_port") or 0),
                external_port=int(ep.get("external_port") or 0),
            )
            for name, ep in (data.get("endpoints") or {}).items()
        }
        return cls(
            id=str(data["id"]),
            provider=data["provider"],
            state=InstanceState(data.get("state", "unknown")),
            created_at=float(data.get("created_at") or time.time()),
            gpu_name=data.get("gpu_name", ""),
            gpu_count=int(data.get("gpu_count") or 0),
            hourly_cost_usd=data.get("hourly_cost_usd"),
            host=data.get("host"),
            ssh_host=data.get("ssh_host"),
            ssh_port=data.get("ssh_port"),
            ssh_user=data.get("ssh_user", "root"),
            endpoints=endpoints,
            label=data.get("label", ""),
            host_id=data.get("host_id"),
        )


@dataclass
class InstanceStatus:
    """What `status` reports after reconciliation (spec section 18)."""

    state: InstanceState
    instance: Optional[Instance] = None
    exists: bool = True
    hourly_cost_usd: Optional[float] = None
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workload:
    """A unit of useful work the user asked for."""

    id: str
    command: list[str]
    label: str = ""
    cwd: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    mode: str = "local"  # local | remote


@dataclass
class Execution:
    id: str
    workload_id: str
    started_at: float
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    state: str = "running"

    @property
    def duration(self) -> float:
        return (self.finished_at or time.time()) - self.started_at


@dataclass
class LogStream:
    """Lazy line iterator over provider logs."""

    lines: Iterator[str]
    source: str = ""

    def __iter__(self) -> Iterator[str]:
        return iter(self.lines)


@dataclass
class ValidationIssue:
    level: str  # error | warning
    message: str
    field: Optional[str] = None


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def error(self, message: str, field: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue("error", message, field))

    def warn(self, message: str, field: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue("warning", message, field))

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        self.issues.extend(other.issues)
        return self
