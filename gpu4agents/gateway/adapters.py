"""Built-in gateway adapters: none, openai, anthropic.

- ``none``      transparent: follow the endpoint, change nothing. For a client
                that already sends the right key and model.
- ``openai``    for OpenAI-shaped clients: drop the client's Authorization,
                inject the instance key, rename the model, normalize effort.
- ``anthropic`` for Claude Code and other Anthropic-wire clients: also drop the
                ``x-api-key`` / ``anthropic-*`` auth so a claude.ai OAuth token
                never reaches the rented machine. Otherwise identical.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.config import GatewayConfig
from .base import GatewayAdapter, Upstream, register


def _normalize_effort(obj: dict[str, Any], policy) -> None:
    """Rewrite a reasoning-effort value in place per the project's policy.

    Handles both the top-level ``reasoning_effort`` and the nested
    ``reasoning: {"effort": ...}`` shape. ``policy`` is a ReasoningEffortPolicy.
    """
    targets: list[tuple[dict, str]] = [(obj, "reasoning_effort")]
    # Anthropic-wire clients (Claude Code) carry the effort under reasoning.effort
    # or output_config.effort, not the OpenAI top-level reasoning_effort.
    for container_key in ("reasoning", "output_config"):
        nested = obj.get(container_key)
        if isinstance(nested, dict):
            targets.append((nested, "effort"))
    for container, key in targets:
        value = container.get(key)
        if not isinstance(value, str):
            continue
        value = policy.map.get(value, value)
        if policy.supported and value not in policy.supported:
            if policy.on_unsupported == "drop":
                container.pop(key, None)
                continue
            if policy.on_unsupported == "error":
                raise ValueError(
                    f"reasoning effort {value!r} not in supported {policy.supported}"
                )
            # "keep": fall through and write the (mapped) value
        container[key] = value


class NoneAdapter(GatewayAdapter):
    name = "none"

    def rewrite_request(self, headers, body, upstream: Upstream):
        return headers, body


class _InjectingAdapter(GatewayAdapter):
    """Shared behavior: inject the instance key, rename the model, normalize effort.

    Subclasses declare which incoming auth headers to strip (``_STRIP``), so a
    client's own credential never reaches the rented machine.
    """

    _STRIP: tuple[str, ...] = ("authorization", "x-api-key")

    def rewrite_request(self, headers, body, upstream: Upstream):
        cfg: GatewayConfig = self.config
        # drop the client's credentials (case-insensitive, exact-match)
        for name in list(headers):
            if name.lower() in self._STRIP:
                headers.pop(name, None)
        if cfg.inject_key and upstream.api_key:
            headers["Authorization"] = f"Bearer {upstream.api_key}"
            headers["x-api-key"] = upstream.api_key
        # mutate the JSON body: model rename + effort normalization
        ctype = next((v for k, v in headers.items() if k.lower() == "content-type"), "")
        if body and ctype.startswith("application/json"):
            try:
                obj = json.loads(body)
            except ValueError:
                return headers, body
            if isinstance(obj, dict):
                if cfg.model_rewrite and "model" in obj and upstream.model:
                    obj["model"] = upstream.model
                _normalize_effort(obj, cfg.reasoning_effort)
                body = json.dumps(obj).encode()
        return headers, body


class OpenAIAdapter(_InjectingAdapter):
    name = "openai"
    _STRIP = ("authorization",)


class AnthropicAdapter(_InjectingAdapter):
    name = "anthropic"
    # Claude Code may carry its claude.ai OAuth token as either header; drop both.
    _STRIP = ("authorization", "x-api-key")

    def rewrite_request(self, headers, body, upstream: Upstream):
        # also drop any anthropic-*key/auth header (prefix match), keeping
        # harmless ones like anthropic-version.
        for name in list(headers):
            low = name.lower()
            if low.startswith("anthropic-") and ("key" in low or "auth" in low):
                headers.pop(name, None)
        return super().rewrite_request(headers, body, upstream)


register("none", lambda: NoneAdapter)
register("openai", lambda: OpenAIAdapter)
register("anthropic", lambda: AnthropicAdapter)
