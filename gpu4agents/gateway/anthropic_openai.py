"""Serve the Anthropic Messages API on top of an OpenAI chat backend.

Claude Code speaks the Anthropic wire protocol (``POST /v1/messages``). vLLM
happens to serve that natively, but llama.cpp — and most other OpenAI-only
backends — do not. This adapter translates the request to OpenAI
``/v1/chat/completions`` and the response back to Anthropic, so any OpenAI
backend can serve Claude Code unchanged.

The translation is split into pure functions (``messages_to_chat``,
``chat_to_message``, and the streaming ``openai_sse_to_anthropic``) so the
mapping is tested in isolation; the adapter just wires them into the gateway's
request/path/response hooks.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Iterator, Optional

from ..core.config import GatewayConfig
from .adapters import _normalize_effort
from .base import GatewayAdapter, Upstream, register

# OpenAI finish_reason -> Anthropic stop_reason.
STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


# --- request: Anthropic /v1/messages -> OpenAI /v1/chat/completions --------

def _system_text(system: Any) -> Optional[str]:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = [b.get("text", "") for b in system
                 if isinstance(b, dict) and b.get("type") == "text"]
        joined = "\n".join(p for p in parts if p)
        return joined or None
    return None


def _image_url(block: dict) -> str:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        return f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    # "url" source (and any other) carries the URL directly.
    return source.get("url", "")


def _tool_result_text(block: dict) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _user_blocks(content: list) -> list[dict]:
    """A user message's content blocks -> OpenAI messages (tool results become
    their own ``tool`` messages; text/image collapse into one user message)."""
    messages: list[dict] = []
    parts: list[dict] = []
    has_image = False
    for b in content:
        t = b.get("type")
        if t == "tool_result":
            messages.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""),
                             "content": _tool_result_text(b)})
        elif t == "text":
            parts.append({"type": "text", "text": b.get("text", "")})
        elif t == "image":
            has_image = True
            parts.append({"type": "image_url", "image_url": {"url": _image_url(b)}})
    if parts:
        if has_image:
            messages.append({"role": "user", "content": parts})
        else:  # text-only collapses to a plain string
            messages.append({"role": "user",
                             "content": "\n".join(p["text"] for p in parts)})
    return messages


def _assistant_blocks(content: list) -> dict:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for b in content:
        t = b.get("type")
        if t == "text":
            text_parts.append(b.get("text", ""))
        elif t == "tool_use":
            tool_calls.append({
                "id": b.get("id", ""),
                "type": "function",
                "function": {"name": b.get("name", ""),
                             "arguments": json.dumps(b.get("input", {}))},
            })
    msg: dict = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return choice
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        return {"type": "function", "function": {"name": choice.get("name", "")}}
    return "auto"


def messages_to_chat(body: dict) -> dict:
    """An Anthropic Messages request body -> an OpenAI chat.completions body."""
    out: dict[str, Any] = {}
    if "model" in body:
        out["model"] = body["model"]
    messages: list[dict] = []
    system = _system_text(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif role == "assistant":
            messages.append(_assistant_blocks(content or []))
        else:  # user (and any non-assistant) role
            messages.extend(_user_blocks(content or []))
    out["messages"] = messages

    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p"):
        if key in body:
            out[key] = body[key]
    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]
    if body.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    if body.get("tools"):
        out["tools"] = [{"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }} for t in body["tools"]]
    if "tool_choice" in body:
        out["tool_choice"] = _tool_choice(body["tool_choice"])
    return out


# --- response: OpenAI chat.completion -> Anthropic message -----------------

def _tool_use_block(call: dict) -> dict:
    try:
        args = json.loads(call.get("function", {}).get("arguments") or "{}")
    except (ValueError, TypeError):
        args = {}
    return {"type": "tool_use", "id": call.get("id", ""),
            "name": call.get("function", {}).get("name", ""), "input": args}


def chat_to_message(resp: dict) -> dict:
    """A non-streaming OpenAI chat.completion -> an Anthropic message."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict] = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        content.append(_tool_use_block(call))
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or "msg_translated",
        "type": "message",
        "role": "assistant",
        "model": resp.get("model", ""),
        "content": content,
        "stop_reason": STOP_REASON.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


def _error_envelope(raw: bytes) -> bytes:
    message, etype = "upstream error", "api_error"
    try:
        obj = json.loads(raw or b"{}")
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict):
            message = err.get("message", message)
            etype = err.get("type", etype)
        elif isinstance(err, str):
            message = err
    except ValueError:
        if raw:
            message = raw.decode(errors="replace")
    return json.dumps({"type": "error", "error": {"type": etype, "message": message}}).encode()


# --- the adapter -----------------------------------------------------------

class AnthropicOpenAIAdapter(GatewayAdapter):
    """Anthropic wire in, OpenAI backend out (and the response back again)."""

    name = "anthropic-openai"
    _STRIP = ("authorization", "x-api-key")

    def rewrite_request(self, headers, body, upstream: Upstream):
        cfg: GatewayConfig = self.config
        # drop the client's credentials so a claude.ai token never leaves here
        for name in list(headers):
            low = name.lower()
            if low in self._STRIP or (low.startswith("anthropic-") and ("key" in low or "auth" in low)):
                headers.pop(name, None)
        if cfg.inject_key and upstream.api_key:
            headers["Authorization"] = f"Bearer {upstream.api_key}"
            headers["x-api-key"] = upstream.api_key

        try:
            anthropic = json.loads(body or b"{}")
        except ValueError:
            return headers, body
        if not isinstance(anthropic, dict):
            return headers, body

        openai = messages_to_chat(anthropic)
        if cfg.model_rewrite and upstream.model:
            openai["model"] = upstream.model
        # Claude Code carries reasoning effort under output_config/reasoning;
        # surface it as OpenAI reasoning_effort, then let the policy map/drop it.
        effort = None
        for key in ("output_config", "reasoning"):
            nested = anthropic.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("effort"), str):
                effort = nested["effort"]
        if effort is not None:
            openai["reasoning_effort"] = effort
            _normalize_effort(openai, cfg.reasoning_effort)
        headers["Content-Type"] = "application/json"
        return headers, json.dumps(openai).encode()

    def target_path(self, path: str) -> str:
        if path.rstrip("/").endswith("/v1/messages"):
            return path.replace("/v1/messages", "/v1/chat/completions")
        return path

    def translate_response(self, request_body, status, headers, chunks):
        out_headers = {k: v for k, v in headers.items() if k.lower() != "content-length"}
        if status >= 400:
            out_headers["Content-Type"] = "application/json"
            return status, out_headers, iter([_error_envelope(b"".join(chunks))])

        streaming = False
        try:
            req = json.loads(request_body or b"{}")
            streaming = bool(req.get("stream"))
        except ValueError:
            pass

        if streaming:
            out_headers["Content-Type"] = "text/event-stream"
            return status, out_headers, openai_sse_to_anthropic(chunks)

        raw = b"".join(chunks)
        try:
            resp = json.loads(raw or b"{}")
        except ValueError:
            return status, out_headers, iter([raw])
        out_headers["Content-Type"] = "application/json"
        return status, out_headers, iter([json.dumps(chat_to_message(resp)).encode()])


# Streaming translation lives in Task 5; defined there.
def openai_sse_to_anthropic(chunks: Iterable[bytes]) -> Iterator[bytes]:  # pragma: no cover
    raise NotImplementedError("streaming translation is implemented in Task 5")


register("anthropic-openai", lambda: AnthropicOpenAIAdapter)
