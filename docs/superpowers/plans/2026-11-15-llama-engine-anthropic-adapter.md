# llama engine + Anthropic-contract adapter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans
> (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Serve GGUF chat models to Claude Code: a first-class `llama`
runtime engine (llama-server) + a gateway that serves the Anthropic
`/v1/messages` contract over any OpenAI backend.

**Architecture:** Three subsystems, built engine-first so the model is
validated over OpenAI `curl` before the translation work. See
`docs/superpowers/specs/2026-11-15-llama-engine-anthropic-adapter-design.md`.

**Tech Stack:** Python 3.10+, stdlib + httpx only (no new deps), pydantic
config, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-llama-engine-anthropic-adapter-design.md`

## Global Constraints

- stdlib + httpx only; no new third-party dependency.
- No secret values in code/config; never printed.
- No AI attribution in commits/PR (owner is sole author).
- Existing adapters (`none`/`openai`/`anthropic`) and the vLLM engine must
  keep passing unchanged except where a test is deliberately updated.

## Review Focus

- **SSE boundary splitting:** an OpenAI chunk may arrive split across TCP
  reads mid-event; the stream translator must buffer by `\n\n`, not assume
  one event per chunk. (Task 5 tests a split-mid-event stream.)
- **Tool-call argument streaming:** OpenAI streams `tool_calls[].function.
  arguments` as concatenated string fragments; Anthropic wants
  `input_json_delta` partial_json. (Task 5 tests fragmented args.)
- **stop_reason mapping when tools are called:** OpenAI `finish_reason:
  "tool_calls"` → Anthropic `stop_reason: "tool_use"`. (Task 4.)
- **Non-JSON / error upstream:** a 4xx/5xx or non-JSON body must become a
  valid Anthropic error envelope, not a crash or a half-translated body.
  (Task 4/5.)
- **llama launcher size:** the packed boot command must stay under the
  provider field cap (Task 2 asserts the start command is bounded).

---

## Task 1: `llama` engine config

**Files:**
- Modify: `gpu4agents/core/config.py` (RuntimeConfigSection)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `runtime.engine == "llama"`; `runtime.quant: Optional[str]`;
  `runtime.n_gpu_layers: Optional[int]` (None = all); `runtime.llama_args:
  dict`.

- [ ] Step 1: Write failing tests: engine accepts `"llama"`; `quant`/
  `n_gpu_layers`/`llama_args` parse with defaults (None/None/{}).
- [ ] Step 2: Run → FAIL (Literal rejects "llama").
- [ ] Step 3: Add `"llama"` to the engine Literal and the three fields.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Commit.

## Task 2: `LlamaRuntime`

**Files:**
- Create: `gpu4agents/runtime/llama.py`
- Modify: `gpu4agents/runtime/__init__.py` (register)
- Test: `tests/test_engine.py` (or `tests/test_llama_runtime.py`)

**Interfaces:**
- Consumes: `ModelRuntime` base, config from Task 1.
- Produces: `LlamaRuntime(engine="llama")` with `server_args()`,
  `start_command()`, `activity_probe()`, `routes()`, `health`,
  `validate()`, `describe()`.

- [ ] Step 1: Write failing tests:
  - `server_args()` includes `-hf <id>:<quant>`, `--host 0.0.0.0`,
    `--port`, `--api-key`, `--metrics`, `--jinja`, `--ctx-size <ctx>`,
    and `--n-gpu-layers` (all → a large int) ;
  - `activity_probe()` returns the `llamacpp:*` gauges/counters;
  - `routes()` == the OpenAI route list;
  - `validate()` errors when `quant` is unset;
  - `describe()` names the model + quant;
  - registered: `runtime_registry.get(..., "llama")` returns it.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement `LlamaRuntime` mirroring `VllmRuntime`; register it.
- [ ] Step 4: Run → PASS (+ full suite green).
- [ ] Step 5: Commit.

## Task 3: Gateway path+response translation capability

**Files:**
- Modify: `gpu4agents/gateway/base.py` (adapter optional hooks)
- Modify: `gpu4agents/gateway/server.py` (call the hooks)
- Test: `tests/test_gateway.py`

**Interfaces:**
- Produces: `GatewayAdapter.target(method, path) -> (method, path)` default
  identity; `GatewayAdapter.translate_response(ctx, status, headers,
  chunks) -> (status, headers, chunks)` default identity; a `ctx` the
  request hook returns for the response hook to read (e.g. stream flag).

- [ ] Step 1: Write failing tests: a stub adapter that rewrites path
  `/a`→`/b` and upper-cases the streamed body reaches the echo upstream at
  `/b` and the client sees the transformed body; default adapters unchanged
  (existing forward test still passes).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Add the optional hooks (default passthrough) and wire the
  server loop to call them; keep chunked streaming.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Commit.

## Task 4: Anthropic→OpenAI request + non-streaming response

**Files:**
- Create: `gpu4agents/gateway/anthropic_openai.py`
- Modify: `gpu4agents/gateway/adapters.py` (register `anthropic-openai`)
- Test: `tests/test_anthropic_openai.py`

**Interfaces:**
- Consumes: Task 3 hooks.
- Produces: adapter `name="anthropic-openai"`; pure functions
  `messages_to_chat(body: dict) -> dict` and `chat_to_message(resp: dict)
  -> dict` for unit testing.

- [ ] Step 1: Write failing tests on the pure functions:
  - system + text/image/tool_use/tool_result blocks → OpenAI messages +
    tool_calls + tool role; tools → OpenAI tools; max_tokens/temp/stop;
  - a chat.completion with text → Anthropic message (end_turn, usage);
  - a chat.completion with tool_calls → `tool_use` block + stop_reason
    `tool_use`;
  - an upstream error body → Anthropic error envelope.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement the mappings + the adapter (non-stream path);
  `target()` sends `/v1/messages`→`/v1/chat/completions`.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Commit.

## Task 5: Streaming translation (OpenAI SSE → Anthropic events)

**Files:**
- Modify: `gpu4agents/gateway/anthropic_openai.py`
- Test: `tests/test_anthropic_openai.py`

**Interfaces:**
- Produces: `openai_sse_to_anthropic(chunks: Iterable[bytes]) ->
  Iterator[bytes]`.

- [ ] Step 1: Write failing tests:
  - a text delta stream → `message_start`, `content_block_start`,
    `content_block_delta`(text_delta)…, `content_block_stop`,
    `message_delta`(end_turn+usage), `message_stop`;
  - a tool-call stream with **fragmented** `arguments` → one tool_use
    block with `input_json_delta` fragments and stop_reason `tool_use`;
  - events split mid-line across chunk boundaries are reassembled.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement the SSE state machine; wire it into
  `translate_response` when the request was `stream: true`.
- [ ] Step 4: Run → PASS (+ full suite green).
- [ ] Step 5: Commit.

## Task 6: End-to-end gateway test + docs

**Files:**
- Test: `tests/test_gateway.py` (fake OpenAI upstream, anthropic-openai)
- Modify: `docs/CONFIG.md` (llama engine + anthropic-openai adapter)

- [ ] Step 1: Write failing test: a fake OpenAI chat upstream; a request to
  the gateway at `/v1/messages` (Anthropic shape) returns an Anthropic
  `message`; a streaming request returns Anthropic SSE events.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Ensure wiring; document both features.
- [ ] Step 4: Run → PASS (+ full suite green).
- [ ] Step 5: Commit.

## Live validation (outside the suite — costs money, user checkpoint)

- Point a project at a GGUF: `engine: llama`, `model.id` =
  `peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF`, `quant: Q5_K_XL`.
- Rent a GPU; `curl` `/v1/chat/completions` (OpenAI) to confirm the model
  answers. **Checkpoint with the user before renting.**
- Set `gateway.adapter: anthropic-openai`; run Claude Code through it;
  confirm a real prompt round-trips and the watchdog counts the traffic.
