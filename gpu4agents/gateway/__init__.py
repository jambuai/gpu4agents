"""Local gateway: a managed loopback endpoint that follows the rented machine.

Importing the package registers the built-in adapters (none/openai/anthropic/
anthropic-openai).
"""

from __future__ import annotations

from .base import GatewayAdapter, Upstream, adapter_names, get_adapter
from .server import EndpointSource, GatewayServer
from . import adapters as _adapters  # noqa: F401 - registers built-ins on import
from . import anthropic_openai as _anthropic_openai  # noqa: F401 - registers anthropic-openai

__all__ = [
    "GatewayAdapter",
    "Upstream",
    "get_adapter",
    "adapter_names",
    "EndpointSource",
    "GatewayServer",
]
