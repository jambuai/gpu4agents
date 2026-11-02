"""The provider contract (spec section 6).

Every provider implements exactly this interface. Provider-specific APIs,
SDKs, identifiers, pricing structures and authentication MUST NOT leak
through it. The core branches on ``capabilities``, never on ``name``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.config import RuntimeConfig
from ..core.credentials import CredentialResolver
from ..core.errors import UnsupportedCapabilityError
from ..core.models import (
    ComputeSpec,
    Execution,
    Instance,
    InstanceStatus,
    LogStream,
    Offer,
    ProviderCapabilities,
    ValidationResult,
    Workload,
)


@dataclass
class WatchdogSpec:
    """What the adapter contributes to the remote lifecycle watchdog.

    The *policy* lives in the core; the adapter only supplies the provider
    calls the watchdog needs to shut its own machine down (spec section 14).
    """

    # Python source executed inside the watchdog. Must define
    #   def provider_stop(ctx) -> None
    #   def provider_destroy(ctx) -> None
    # using the stdlib only.
    actions_source: str
    # Environment variables the instance needs for those calls (credentials).
    env: dict[str, str] = field(default_factory=dict)
    # Static values available to the actions through ctx["provider"].
    context: dict[str, Any] = field(default_factory=dict)


class GPUProvider(ABC):
    """Stable internal contract implemented by every adapter."""

    name: str = "abstract"
    display_name: str = "Abstract"
    required_credentials: list[str] = []
    capabilities: ProviderCapabilities = ProviderCapabilities()
    # Runtime engine -> the image to use when compute.image is unset. A
    # provider names the copy its hosts already cache, so a first boot does
    # not wait on a multi-GB registry pull.
    preferred_images: dict[str, str] = {}

    def __init__(self, config: RuntimeConfig, credentials: CredentialResolver) -> None:
        self.config = config
        self.credentials = credentials

    # -- contract -----------------------------------------------------------

    @abstractmethod
    def validate(self, config: RuntimeConfig) -> ValidationResult:
        """Check credentials, capabilities and the requested spec. No side effects."""

    @abstractmethod
    def setup(self, spec: ComputeSpec) -> Instance:
        """Provision compute matching ``spec`` and return a normalized handle."""

    @abstractmethod
    def start(self, instance: Instance) -> Instance:
        """Bring a stopped instance back to running."""

    @abstractmethod
    def run(self, instance: Instance, workload: Workload) -> Execution:
        """Execute a workload on the instance (requires ``capabilities.remote_exec``)."""

    @abstractmethod
    def status(self, instance: Instance) -> InstanceStatus:
        """Observed provider-side state, normalized."""

    @abstractmethod
    def stop(self, instance: Instance) -> None:
        """Stop compute. MUST be idempotent (spec section 23)."""

    @abstractmethod
    def destroy(self, instance: Instance) -> None:
        """Destroy the remote resource. MUST be idempotent."""

    @abstractmethod
    def logs(self, instance: Instance) -> LogStream:
        """Provider-side logs for the instance."""

    # -- optional, capability-gated ----------------------------------------

    def find_offers(self, spec: ComputeSpec, limit: int = 10) -> list[Offer]:
        """Discover purchasable capacity (requires ``capabilities.pricing_query``)."""
        raise UnsupportedCapabilityError(f"{self.name} cannot query offers")

    def list_instances(self) -> list[Instance]:
        """Every instance of this tool in the account (requires ``capabilities.list_instances``).

        Returns only instances whose label starts with ``JAMBU_LABEL_PREFIX``;
        anything else in the account is not ours to report or touch.
        """
        raise UnsupportedCapabilityError(f"{self.name} cannot list instances")

    def watchdog_spec(self, spec_or_instance: Any) -> Optional[WatchdogSpec]:
        """Provider calls the remote watchdog uses to stop/destroy itself."""
        return None

    def refresh(self, instance: Instance) -> Instance:
        """Re-read provider data into the normalized handle (ports, ip, state)."""
        status = self.status(instance)
        return status.instance or instance

    # -- helpers ------------------------------------------------------------

    def require_capability(self, name: str) -> None:
        if not getattr(self.capabilities, name, False):
            raise UnsupportedCapabilityError(
                f"provider '{self.name}' does not support capability '{name}'"
            )

    def credential_status(self) -> dict[str, bool]:
        return {name: bool(self.credentials.get(name)) for name in self.required_credentials}

    def is_configured(self) -> bool:
        return all(self.credential_status().values()) if self.required_credentials else True

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} name={self.name!r}>"
