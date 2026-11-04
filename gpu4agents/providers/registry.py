"""Provider registry (spec section 20).

Adding a provider means writing one class against the contract in ``base.py``.
No lifecycle, execution or CLI code changes. A provider reaches the registry
in one of three ways:

- built in: registered by ``load_builtin_providers`` (Vast.ai);
- installed: a package that declares a ``gpu4agents.providers`` entry point;
- per project: ``provider.plugin`` in gpu.yaml, a file or module path.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterator, Optional, Type

from ..core.config import RuntimeConfig
from ..core.credentials import CredentialResolver
from ..core.errors import ConfigError
from .base import GPUProvider

ENTRY_POINT_GROUP = "gpu4agents.providers"


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
            raise ConfigError(
                f"unknown provider '{name}' (available: {known}). For another provider, "
                "write an adapter: gpu providers new <name>"
            ) from None

    def class_for(self, config: RuntimeConfig) -> Type[GPUProvider]:
        """The adapter a config selects, loading its plugin first if it names one."""
        if config.provider.plugin:
            cls = load_plugin(config.provider.plugin, config.project_dir)
            if cls.name != config.provider.name:
                raise ConfigError(
                    f"provider.plugin {config.provider.plugin} defines provider "
                    f"'{cls.name}', but provider.name is '{config.provider.name}'; "
                    "make them the same"
                )
            self.register(cls)
        return self.get_class(config.provider.name)

    def get(
        self,
        config: RuntimeConfig,
        credentials: Optional[CredentialResolver] = None,
        name: Optional[str] = None,
    ) -> GPUProvider:
        if name and name != config.provider.name:
            provider_cls = self.get_class(name)
        else:
            provider_cls = self.class_for(config)
        resolver = credentials or CredentialResolver(
            config.project_dir, profile=config.provider.profile
        )
        return provider_cls(config, resolver)


registry = ProviderRegistry()


def load_plugin(spec: str, base_dir: Optional[Path] = None) -> Type[GPUProvider]:
    """Import ``path/to/file.py:Class`` or ``package.module:Class``."""
    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise ConfigError(
            f"provider.plugin must look like ./file.py:ClassName or module:ClassName, got {spec!r}"
        )
    if target.endswith(".py") or "/" in target or "\\" in target:
        path = Path(target).expanduser()
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        if not path.is_file():
            raise ConfigError(f"provider.plugin: {path} does not exist")
        module_name = f"gpu4agents_plugin_{path.stem}_{abs(hash(str(path.resolve())))}"
        module = sys.modules.get(module_name)
        if module is None:
            loader_spec = importlib.util.spec_from_file_location(module_name, path)
            if loader_spec is None or loader_spec.loader is None:
                raise ConfigError(f"provider.plugin: cannot load {path}")
            module = importlib.util.module_from_spec(loader_spec)
            sys.modules[module_name] = module
            try:
                loader_spec.loader.exec_module(module)
            except Exception as exc:
                sys.modules.pop(module_name, None)
                raise ConfigError(f"provider.plugin: importing {path} failed: {exc!r}") from exc
    else:
        try:
            module = importlib.import_module(target)
        except Exception as exc:
            raise ConfigError(f"provider.plugin: importing {target} failed: {exc!r}") from exc
    cls = getattr(module, attr, None)
    if not (isinstance(cls, type) and issubclass(cls, GPUProvider)):
        raise ConfigError(f"provider.plugin: {spec} is not a GPUProvider subclass")
    return cls


def _load_entry_points() -> None:
    try:
        found = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # Python 3.9 shape, kept for safety
        found = metadata.entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[union-attr]
    for entry in found:
        try:
            cls = entry.load()
        except Exception:  # a broken third-party package must not break the CLI
            continue
        if isinstance(cls, type) and issubclass(cls, GPUProvider):
            registry.register(cls)


def load_builtin_providers() -> ProviderRegistry:
    """Register the adapters that ship with the CLI, then installed ones."""
    from .vast.provider import VastProvider  # noqa: WPS433 (deferred import by design)

    registry.register(VastProvider)
    _load_entry_points()
    return registry
