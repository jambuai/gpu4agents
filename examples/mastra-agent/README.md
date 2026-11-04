# Mastra agent example

A real, installable [Mastra](https://mastra.ai) agent project — not a hand-rolled
tool-calling loop — running against the vLLM instance `gpu` provisioned for
you. gpu builds no agent runtime; this is the actual Mastra pattern from
[their own docs on tools](https://github.com/mastra-ai/mastra/blob/main/docs/src/content/en/docs/agents/tools.mdx),
pointed at a self-hosted endpoint instead of the real OpenAI API.

## What connects it to gpu

`src/agent.ts` reads three env vars — nothing else:

| Env var | Set by |
| --- | --- |
| `JAMBU_GPU_ENDPOINT` | `gpu run` |
| `JAMBU_MODEL_ID` | `gpu run` |
| `OPENAI_API_KEY` | `gpu run` (the key generated for the instance, or `runtime.api_key`) |

That's the entire integration surface. No gpu SDK, no wrapper client.

## Requirements on the gpu.yaml side

Mastra calls `weatherTool` through vLLM's regular OpenAI tool-calling — which only
fires if vLLM was started with a matching `--tool-call-parser`. gpu picks one
automatically for recognized model families (`runtime.tool_calling: auto`, the
default); for anything it doesn't recognize, set it explicitly:

```yaml
# gpu.yaml
gpu_runtime:
  model:
    id: Qwen/Qwen2.5-7B-Instruct   # gpu resolves "hermes" for this one

  runtime:
    engine: vllm
    tool_calling: auto             # or: tool_call_parser: hermes
```

Run `gpu validate` first — it warns if no parser could be resolved for your
`model.id`, which means the agent below would run but the model would never
actually be able to call `weatherTool`.

## Run it

```bash
cd examples/mastra-agent
npm install
cd ../..                    # back to wherever gpu.yaml lives

gpu setup
gpu run npm --prefix examples/mastra-agent start
```

`gpu run` guarantees the instance is READY and injects
`JAMBU_GPU_ENDPOINT` / `JAMBU_MODEL_ID` / `OPENAI_API_KEY` into that `npm start` —
the same as it would for any other workload. You should see the model call
`weatherTool` and answer with a real current-conditions sentence, e.g.:

```
It's currently 18°C and partly cloudy in Lisbon.
```

## Switching to the Responses API

`src/agent.ts` uses `api: "chat"` (vLLM's `/v1/chat/completions`) because it works
on every vLLM version. vLLM also serves the OpenAI **Responses API** at
`/v1/responses`, including MCP tool routing done server-side — set
`api: "responses"` in the model config and `runtime.tool_server` in `gpu.yaml`
(see [`docs/CONFIG.md`](../../docs/CONFIG.md#agentic-flows-nothing-to-build)) to
use that instead.
