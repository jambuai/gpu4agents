"""Local gateway: a managed loopback endpoint that follows the rented machine.

Importing the package registers the built-in adapters (none/openai/anthropic).
"""

from __future__ import annotations

from .base import GatewayAdapter, Upstream, adapter_names, get_adapter
from .server import EndpointSource, GatewayServer
from . import adapters as _adapters  # noqa: F401 - registers built-ins on import

__all__ = [
    "GatewayAdapter",
    "Upstream",
    "get_adapter",
    "adapter_names",
    "EndpointSource",
    "GatewayServer",
]
