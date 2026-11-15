# Design: `llama` serving engine + Anthropic-contract gateway adapter

**Date:** 2026-11-15
**Status:** awaiting approval

## Intent / why

gpu4agents serves chat models on **vLLM**, which needs safetensors weights. A
large and growing class of useful models — abliterated / uncensored coders,
cyber-tuned MoEs (the trigger here is `Cyber-Tiel-Coder-35B-A3B`) — ship
**only as GGUF** (llama.cpp) and MLX (Apple). vLLM cannot serve them cleanly.

Separately, the primary agent client is **Claude Code**, which speaks the
**Anthropic Messages API** (`POST /v1/messages`). Today that works *only*
because vLLM happens to serve `/v1/messages` natively. **Any OpenAI-only
backend — llama.cpp and most others — cannot serve Claude Code without a
translation layer.**

Two gaps, one goal: **let gpu4agents serve GGUF models to Claude Code**, and in
doing so give every future OpenAI-only backend a reusable Anthropic front door.

## Goals

- **G1 — `llama` engine.** A first-class runtime engine that serves any GGUF
  chat model from Hugging Face via `llama-server`, OpenAI-native, under the
  same lifecycle contract as vLLM: `/health`, a Prometheus activity probe the
  watchdog reads, per-instance API key, boot script with network-error
  restarts, declared routes.
- **G2 — Anthropic over OpenAI.** The gateway can *serve the Anthropic
  contract on top of an OpenAI backend*: translate the `/v1/messages` request
  into OpenAI `/v1/chat/completions`, and translate the response — both
  non-streaming and SSE — back to Anthropic wire format, including
  `tool_use` / `tool_result` and token `usage`.
- **G3 — End to end.** Claude Code → `gpu` gateway (Anthropic-translating
  adapter) → `llama` engine (GGUF) → answer. The watchdog counts the traffic,
  so the machine stays up while in use and idles out when not (the exact
  property we verified was missing in a naive llama.cpp setup).

## Non-goals

- **Not** replacing vLLM. vLLM stays the default; its native `/v1/messages`
  path keeps working with **no translation** (the `anthropic` passthrough
  adapter stays as is).
- **Not** MLX (Apple-silicon only).
- **Not** re-implementing Anthropic features that do not map to OpenAI chat.
  Anything with no clean mapping **fails explicitly**, never silently wrong.
- **Not** MTP / speculative GGUF variants in v1 — use the plain GGUF.
- **Not** auto-selecting the `llama` engine from a model card in v1 (the user
  sets `engine: llama`); card-based auto-detection can come later.

## Global constraints (from the repo and CLAUDE.md)

- **Dependencies:** stdlib + `httpx` only (httpx already required). No new
  third-party dependency — the translation is hand-rolled.
- **Boot field size cap** (Vast ≈ 16 KB): reuse vLLM's base85+zlib launcher
  packing if the llama launcher approaches it.
- **Secrets:** never in `gpu.yaml`; provider key via `gpu login`; `HF_TOKEN`
  via env for gated repos. Never printed.
- **Authorship:** every commit / PR is the owner's; **no** `Co-Authored-By`,
  `Claude-Session`, "Generated with Claude Code" or any AI attribution.

## Approaches considered

**A. One translating gateway adapter, hand-rolled (CHOSEN).** Extend the
existing gateway adapter contract so an adapter may also rewrite the upstream
path/method and translate the response stream; add an Anthropic→OpenAI adapter
on top. One gateway, one selection knob (`gateway.adapter`), one plugin
mechanism, zero new deps.

**B. Depend on an existing shim** (LiteLLM / `claude-code-proxy` / y-router).
Rejected: pulls a heavy dependency into a deliberately minimal tool, and still
has to be wired into the gateway/lifecycle/secret model. We lose control over
Claude Code's wire quirks we would then be debugging through someone else's
abstraction.

**C. A separate translating proxy process** beside the gateway. Rejected:
duplicates transport, endpoint-refresh and credential handling that the
gateway already owns; two things to run and supervise instead of one.

**llama via the `custom` engine** (a hand-written server dir) was the original
fallback. Rejected as the product answer: it buries a reusable capability in
one project's `server/` dir. A first-class `llama` engine serves every future
GGUF model.

## Architecture — three subsystems (→ three plans)

### Subsystem 3 — `llama` runtime engine *(built first)*

Mirror `VllmRuntime` (`runtime/vllm.py`):

- `engine = "llama"`; add `"llama"` to `RuntimeConfigSection.engine` Literal.
- **Image:** official CUDA server image `ghcr.io/ggml-org/llama.cpp:server-cuda`,
  pinned to a specific tag/digest (never `latest`), with CUDA / compute-cap
  requirements declared like vLLM's.
- **Launcher:** `llama-server -hf <repo>:<QUANT> --host 0.0.0.0 --port $PORT
  --n-gpu-layers <N|all> --ctx-size <context_length> --api-key <key>
  --metrics --jinja [--flash-attn]`, wrapped in the same network-restart bash
  the vLLM launcher uses.
- **activity_probe:** `kind: prometheus`,
  gauges `["llamacpp:requests_processing", "llamacpp:requests_deferred"]`,
  counters `["llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total"]`.
- **health:** `/health` (200 ready / 503 loading).
- **routes:** `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/models`.
- **Config fields:** `model.id` = GGUF repo id; new `runtime.quant` (HF quant
  tag, e.g. `Q5_K_XL`); `runtime.n_gpu_layers` (default: all);
  `runtime.llama_args` escape hatch (mirrors `vllm_args`, flattened the same
  way). `context_length` → `--ctx-size`; `dtype`/`gpu_memory_utilization`/
  `tensor_parallel_size` are vLLM-only and ignored here.
- **validate():** quant set; `model.id` looks like a HF repo; warn if the repo
  name has no GGUF marker.

Delivered first so the actual model is validated with a plain **OpenAI `curl`**
before any translation work.

### Subsystem 1 — Gateway: path + response translation capability

Today (`gateway/base.py`, `gateway/server.py`): an adapter only does
`rewrite_request(headers, body, upstream) -> (headers, body)`; the server
forwards `self.path` verbatim and streams `resp.iter_raw()` through untouched.

Extend the adapter contract with **two optional hooks**, defaulting to today's
passthrough so `none`/`openai`/`anthropic` are unchanged:

- `target(method, path) -> (method, path)` — choose the upstream method/path
  (default: unchanged).
- `translate_response(request_ctx, status, headers, chunks) -> (status,
  headers, chunks)` — `chunks` is an iterator of upstream body bytes; returns
  the client-facing status/headers and a (possibly re-generated) byte-chunk
  iterator. Default: identity. Streaming is preserved because the translator
  is a **generator** over upstream chunks.

The server loop calls `rewrite_request` (as now), then `target(...)` to build
the URL, then wraps the streamed response through `translate_response`.

### Subsystem 2 — Anthropic↔OpenAI translating adapter *(built on #1)*

A new adapter (`anthropic-openai`, selected via `gateway.adapter`) that turns a
Claude Code request into OpenAI chat and the answer back:

- **Request → OpenAI:** `/v1/messages` → `/v1/chat/completions`; `system` →
  leading system message; `messages` content blocks (`text`, `image`,
  `tool_use`, `tool_result`) → OpenAI messages + `tool_calls` + `tool` role;
  `tools` → OpenAI `tools`; `max_tokens`, `temperature`, `top_p`, `stop`,
  `stream`; reasoning effort already normalized by the base injector.
- **Response (non-stream) → Anthropic:** `chat.completion` → a `message` with
  content blocks, `tool_use` blocks from `tool_calls`, `stop_reason` mapping
  (`stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`), and
  `usage` from OpenAI `usage`.
- **Response (SSE) → Anthropic events:** OpenAI delta stream → the Anthropic
  event sequence `message_start` → `content_block_start` /
  `content_block_delta` (`text_delta`, and `input_json_delta` for tool args) /
  `content_block_stop` → `message_delta` (stop_reason + usage) →
  `message_stop`, with periodic `ping`.
- **Errors** mapped to the Anthropic error envelope.

Reusable for **any** OpenAI backend (llama *or* vLLM's OpenAI routes).

## Order of work

1. **Plan 3 — `llama` engine.** Ship + rent a GPU + validate the GGUF model
   over OpenAI `curl`. Checkpoint: the model answers.
2. **Plan 1 — gateway capability.** Path + streaming response translation
   hooks, passthrough-safe for existing adapters.
3. **Plan 2 — Anthropic adapter.** Full request/response/stream translation.
4. **End to end:** point `local-cc` / Claude Code at the gateway and confirm a
   real prompt round-trips, with the watchdog counting it as activity.

Each plan produces working, tested software on its own (TDD throughout).

## What is left to the builder (resolved in the plans)

- Exact new config field names and defaults.
- The pinned llama.cpp image tag/digest and its CUDA/compute-cap floor.
- The field-by-field OpenAI⇄Anthropic mapping tables per content-block type,
  with test fixtures (captured from a real Claude Code request + a real
  llama-server response).
- Whether the Anthropic adapter is a new name or a mode of `anthropic`.
