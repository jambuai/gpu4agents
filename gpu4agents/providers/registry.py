"""Provider registry (spec section 20).

Adding a provider means adding ``providers/<name>/`` and registering the class.
No lifecycle, execution or CLI code changes.
"""

from __future__ import annotations

from typing import Iterator, Optional, Type

from ..core.config import RuntimeConfig
from ..core.credentials import CredentialResolver
from ..core.errors import ConfigError
from .base import GPUProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Type[GPUProvider]] = {}

    def register(self, provider_cls: Type[GPUProvider]) -> Type[GPUProvider]:
        name = provider_cls.name
        if not name or name == "abstract":
            raise ValueError(f"{provider_cls!r} must define a provider name")
        self._providers[name] = provider_cls
        return provider_cls

    def names(self) -> list[str]:
        return sorted(self._providers)

    def classes(self) -> Iterator[Type[GPUProvider]]:
        for name in self.names():
            yield self._providers[name]

    def get_class(self, name: str) -> Type[GPUProvider]:
        try:
            return self._providers[name]
        except KeyError:
            known = ", ".join(self.names()) or "<none>"
            raise ConfigError(f"unknown provider '{name}' (available: {known})") from None

    def get(
        self,
        config: RuntimeConfig,
        credentials: Optional[CredentialResolver] = None,
        name: Optional[str] = None,
    ) -> GPUProvider:
        provider_cls = self.get_class(name or config.provider.name)
        resolver = credentials or CredentialResolver(
            config.project_dir, profile=config.provider.profile
        )
        return provider_cls(config, resolver)


registry = ProviderRegistry()


def load_builtin_providers() -> ProviderRegistry:
    """Import and register every adapter shipped with the CLI."""
    from .vast.provider import VastProvider  # noqa: WPS433 (deferred import by design)

    registry.register(VastProvider)
    return registry
