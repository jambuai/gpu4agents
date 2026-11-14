"""Gateway adapters: how a request is shaped for one client before it is forwarded.

An adapter is the per-client half of the local gateway. The gateway owns the
transport (a loopback HTTP server that follows the rented machine's endpoint);
the adapter owns the request shaping: which of the client's headers to drop,
whether to inject the instance key, whether to rename the model, how to
reconcile reasoning effort. The core forwards; the adapter decides.

The interface is deliberately small and request-only. Responses (including
streaming ones) pass straight through, so an adapter never has to deal with
chunked bodies. Built-ins live in ``adapters.py``; a project can point
``gateway.plugin`` at its own class the same way a provider plugin works.
"""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from ..core.config import GatewayConfig
from ..core.errors import JambuError


class Upstream:
    """What the adapter needs to know about the machine it is forwarding to.

    Mirrors the fields of ``Engine.connection``: ``endpoint`` (raw host:port),
    ``base_url`` (``.../v1``), ``api_key`` (the per-instance vLLM key) and
    ``model`` (the served model id).
    """

    __slots__ = ("endpoint", "base_url", "api_key", "model")

    def __init__(self, endpoint: str, base_url: Optional[str], api_key: Optional[str], model: Optional[str]) -> None:
        self.endpoint = endpoint
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_connection(cls, conn: dict) -> "Upstream":
        return cls(conn.get("endpoint"), conn.get("base_url"), conn.get("api_key"), conn.get("model"))


class GatewayAdapter(ABC):
    """Shapes one request for one kind of client. Stateless apart from its config."""

    #: must equal the ``gateway.adapter`` value that selects it
    name: str = ""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config

    @abstractmethod
    def rewrite_request(
        self, headers: dict[str, str], body: bytes, upstream: Upstream
    ) -> tuple[dict[str, str], bytes]:
        """Return the (headers, body) to forward upstream.

        ``headers`` is a fresh mutable copy of the client's headers with hop-by-hop
        headers already removed; the adapter may add, drop or overwrite entries.
        ``body`` is the raw request body (possibly empty).
        """
        raise NotImplementedError


# --- registry -------------------------------------------------------------

_BUILTINS: dict[str, Callable[[], type[GatewayAdapter]]] = {}


def register(name: str, loader: Callable[[], type[GatewayAdapter]]) -> None:
    _BUILTINS[name] = loader


def _load_plugin(spec: str) -> type[GatewayAdapter]:
    """Resolve ``./file.py:Class`` (relative to cwd) or ``module.path:Class``."""
    if ":" not in spec:
        raise JambuError(f"gateway.plugin must be 'target:Class', got {spec!r}")
    target, _, class_name = spec.partition(":")
    if target.endswith(".py") or "/" in target:
        path = Path(target).expanduser().resolve()
        if not path.is_file():
            raise JambuError(f"gateway.plugin file not found: {path}")
        mod_spec = importlib.util.spec_from_file_location(f"_gw_plugin_{path.stem}", path)
        module = importlib.util.module_from_spec(mod_spec)  # type: ignore[arg-type]
        mod_spec.loader.exec_module(module)  # type: ignore[union-attr]
    else:
        module = importlib.import_module(target)
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, GatewayAdapter):
        raise JambuError(f"gateway.plugin {spec!r} is not a GatewayAdapter subclass")
    return cls


def get_adapter(config: GatewayConfig) -> GatewayAdapter:
    """Build the adapter named by ``config.adapter`` (or its ``plugin``)."""
    if config.plugin:
        return _load_plugin(config.plugin)(config)
    loader = _BUILTINS.get(config.adapter)
    if loader is None:
        known = ", ".join(sorted(_BUILTINS)) or "(none)"
        raise JambuError(f"unknown gateway adapter {config.adapter!r}; known: {known}")
    return loader()(config)


def adapter_names() -> list[str]:
    return sorted(_BUILTINS)
