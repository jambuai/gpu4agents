"""The Anthropic-over-OpenAI gateway adapter: request + response translation."""

from __future__ import annotations

import json

import pytest

from gpu4agents.core.config import GatewayConfig
from gpu4agents.gateway import get_adapter
from gpu4agents.gateway.base import Upstream
from gpu4agents.gateway.anthropic_openai import chat_to_message, messages_to_chat

UP = Upstream("http://1.2.3.4:40301", "http://1.2.3.4:40301/v1", "SECRETKEY", "served/Model")


# --- request: messages_to_chat --------------------------------------------

def test_system_prompt_becomes_system_message():
    out = messages_to_chat({
        "model": "m", "max_tokens": 100,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert out["max_tokens"] == 100


def test_system_as_block_list_is_joined():
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        "messages": [{"role": "user", "content": "x"}],
    })
    assert out["messages"][0] == {"role": "system", "content": "A\nB"}


def test_text_blocks_collapse_to_string():
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    })
    assert out["messages"][0] == {"role": "user", "content": "hello"}


def test_image_block_becomes_image_url():
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
        ]}],
    })
    parts = out["messages"][0]["content"]
    assert {"type": "text", "text": "look"} in parts
    assert any(p.get("type") == "image_url" and
               p["image_url"]["url"] == "data:image/png;base64,QUJD" for p in parts)


def test_assistant_tool_use_becomes_tool_calls():
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "messages": [{"role": "assistant", "content": [
            {"type": "text", "text": "I'll run it"},
            {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}},
        ]}],
    })
    msg = out["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["id"] == "toolu_1"
    assert msg["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"cmd": "ls"}


def test_user_tool_result_becomes_tool_message():
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file.txt"},
        ]}],
    })
    msg = out["messages"][0]
    assert msg == {"role": "tool", "tool_call_id": "toolu_1", "content": "file.txt"}


def test_tool_result_block_list_is_joined():
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]},
        ]}],
    })
    assert out["messages"][0]["content"] == "line1\nline2"


def test_tools_and_tool_choice_mapping():
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
    })
    assert out["tools"][0] == {"type": "function", "function": {
        "name": "bash", "description": "run", "parameters": {"type": "object"}}}
    assert out["tool_choice"] == "required"


def test_tool_choice_specific_tool():
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "tool", "name": "bash"},
    })
    assert out["tool_choice"] == {"type": "function", "function": {"name": "bash"}}


def test_sampling_params_and_stream_options():
    out = messages_to_chat({
        "model": "m", "max_tokens": 50, "temperature": 0.7, "top_p": 0.9,
        "stop_sequences": ["X"], "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["temperature"] == 0.7 and out["top_p"] == 0.9
    assert out["stop"] == ["X"]
    assert out["stream"] is True
    assert out["stream_options"] == {"include_usage": True}


# --- response: chat_to_message --------------------------------------------

def test_chat_completion_text_to_anthropic_message():
    msg = chat_to_message({
        "id": "chatcmpl-1", "model": "served/Model",
        "choices": [{"message": {"role": "assistant", "content": "hello there"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    })
    assert msg["type"] == "message" and msg["role"] == "assistant"
    assert msg["content"] == [{"type": "text", "text": "hello there"}]
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"] == {"input_tokens": 10, "output_tokens": 3}


def test_chat_completion_length_maps_to_max_tokens():
    msg = chat_to_message({"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]})
    assert msg["stop_reason"] == "max_tokens"


def test_chat_completion_tool_calls_to_tool_use():
    msg = chat_to_message({
        "id": "chatcmpl-2", "model": "m",
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "bash", "arguments": '{"cmd":"ls"}'}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    })
    tool_blocks = [b for b in msg["content"] if b["type"] == "tool_use"]
    assert tool_blocks[0] == {"type": "tool_use", "id": "call_1", "name": "bash",
                              "input": {"cmd": "ls"}}
    assert msg["stop_reason"] == "tool_use"


# --- adapter: path + request rewrite + non-streaming response --------------

def _adapter():
    return get_adapter(GatewayConfig(adapter="anthropic-openai"))


def test_adapter_registered():
    from gpu4agents.gateway import adapter_names
    assert "anthropic-openai" in adapter_names()


def test_adapter_rewrites_path():
    assert _adapter().target_path("/v1/messages") == "/v1/chat/completions"
    assert _adapter().target_path("/v1/models") == "/v1/models"


def test_adapter_request_translates_injects_key_and_renames_model():
    a = _adapter()
    body = json.dumps({"model": "claude-x", "max_tokens": 10,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    headers = {"x-api-key": "LEAK", "Authorization": "Bearer LEAK", "Content-Type": "application/json"}
    out_h, out_b = a.rewrite_request(headers, body, UP)
    obj = json.loads(out_b)
    assert obj["model"] == "served/Model"                 # renamed to the served model
    assert obj["messages"][0] == {"role": "user", "content": "hi"}
    assert out_h["Authorization"] == "Bearer SECRETKEY"   # instance key injected
    assert "LEAK" not in json.dumps(out_h)


def test_adapter_nonstream_response_to_anthropic():
    a = _adapter()
    req = json.dumps({"stream": False, "messages": []}).encode()
    openai = json.dumps({"id": "c", "model": "m",
                         "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
    status, headers, chunks = a.translate_response(req, 200, {"Content-Type": "application/json"},
                                                   iter([openai]))
    out = json.loads(b"".join(chunks))
    assert status == 200
    assert out["type"] == "message"
    assert out["content"] == [{"type": "text", "text": "hi"}]


def test_adapter_upstream_error_becomes_anthropic_error():
    a = _adapter()
    req = json.dumps({"stream": False}).encode()
    err = json.dumps({"error": {"message": "bad", "type": "invalid_request_error"}}).encode()
    status, headers, chunks = a.translate_response(req, 400, {"Content-Type": "application/json"},
                                                   iter([err]))
    out = json.loads(b"".join(chunks))
    assert status == 400
    assert out["type"] == "error"
    assert out["error"]["message"] == "bad"


# --- Task 5: streaming OpenAI SSE -> Anthropic events ----------------------

from gpu4agents.gateway.anthropic_openai import openai_sse_to_anthropic  # noqa: E402


def _parse_anthropic_sse(raw: bytes):
    """-> list of (event_name, data_dict) from an Anthropic SSE byte stream."""
    events = []
    for block in raw.split(b"\n\n"):
        block = block.strip()
        if not block:
            continue
        name = None
        data = None
        for line in block.split(b"\n"):
            if line.startswith(b"event:"):
                name = line[6:].strip().decode()
            elif line.startswith(b"data:"):
                data = json.loads(line[5:].strip())
        events.append((name, data))
    return events


def _sse(*objs):
    return [f"data: {json.dumps(o)}\n\n".encode() for o in objs] + [b"data: [DONE]\n\n"]


def test_stream_text_produces_anthropic_event_sequence():
    chunks = _sse(
        {"id": "c1", "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"id": "c1", "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
        {"id": "c1", "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
        {"id": "c1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "c1", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
    )
    events = _parse_anthropic_sse(b"".join(openai_sse_to_anthropic(iter(chunks))))
    names = [n for n, _ in events]
    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert names[-1] == "message_stop"
    assert names[-2] == "message_delta"
    # the text deltas reassemble
    text = "".join(d["delta"]["text"] for n, d in events
                    if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert text == "Hello"
    msg_delta = [d for n, d in events if n == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "end_turn"
    assert msg_delta["usage"]["output_tokens"] == 2


def test_stream_tool_call_with_fragmented_arguments():
    chunks = _sse(
        {"id": "c2", "model": "m", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "bash", "arguments": ""}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"cmd":'}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"ls"}'}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    )
    events = _parse_anthropic_sse(b"".join(openai_sse_to_anthropic(iter(chunks))))
    starts = [d for n, d in events if n == "content_block_start"]
    assert starts[0]["content_block"] == {"type": "tool_use", "id": "call_1", "name": "bash", "input": {}}
    partial = "".join(d["delta"]["partial_json"] for n, d in events
                      if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
    assert partial == '{"cmd":"ls"}'
    msg_delta = [d for n, d in events if n == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


def test_stream_reassembles_events_split_across_chunks():
    full = b"".join(_sse(
        {"id": "c3", "model": "m", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
        {"id": "c3", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ))
    # chop the byte stream into tiny pieces, splitting events mid-line
    pieces = [full[i:i + 7] for i in range(0, len(full), 7)]
    events = _parse_anthropic_sse(b"".join(openai_sse_to_anthropic(iter(pieces))))
    text = "".join(d["delta"]["text"] for n, d in events
                    if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert text == "Hello"
    assert [n for n, _ in events][-1] == "message_stop"


# --- review fixes: assistant content fallback, stream usage, 200-error -----

def test_assistant_with_only_unknown_blocks_gets_empty_string_content():
    # e.g. an extended-thinking replay: assistant message with only a thinking
    # block. OpenAI backends reject content:null with no tool_calls -> use "".
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm"}]}],
    })
    assert out["messages"][0] == {"role": "assistant", "content": ""}


def test_assistant_tool_use_only_keeps_null_content():
    # with tool_calls, null content is legal and must be preserved
    out = messages_to_chat({
        "model": "m", "max_tokens": 1,
        "messages": [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "x", "input": {}}]}],
    })
    assert out["messages"][0]["content"] is None
    assert out["messages"][0]["tool_calls"][0]["id"] == "t"


def test_stream_message_delta_carries_input_tokens():
    chunks = _sse(
        {"id": "c", "model": "m", "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]},
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "c", "choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 1}},
    )
    events = _parse_anthropic_sse(b"".join(openai_sse_to_anthropic(iter(chunks))))
    msg_delta = [d for n, d in events if n == "message_delta"][0]
    assert msg_delta["usage"]["input_tokens"] == 42
    assert msg_delta["usage"]["output_tokens"] == 1


def test_nonstream_200_with_error_body_becomes_anthropic_error():
    a = _adapter()
    req = json.dumps({"stream": False}).encode()
    body = json.dumps({"error": {"message": "boom", "type": "server_error"}}).encode()
    status, headers, chunks = a.translate_response(req, 200, {"Content-Type": "application/json"},
                                                   iter([body]))
    out = json.loads(b"".join(chunks))
    assert out["type"] == "error"
    assert out["error"]["message"] == "boom"


# --- fix: Claude Code sends /v1/messages?beta=true (query string) -----------

def test_target_path_rewrites_with_query_string():
    # Claude Code always calls /v1/messages?beta=true; the query must not break
    # the rewrite, and it must be preserved on the rewritten path.
    a = _adapter()
    assert a.target_path("/v1/messages?beta=true") == "/v1/chat/completions?beta=true"
    assert a.target_path("/v1/messages/?beta=true") == "/v1/chat/completions?beta=true"


def test_target_path_plain_and_other_unchanged():
    a = _adapter()
    assert a.target_path("/v1/messages") == "/v1/chat/completions"
    assert a.target_path("/v1/models") == "/v1/models"
    assert a.target_path("/v1/models?x=1") == "/v1/models?x=1"
