# Using a self-hosted model with Claude Code

`gpu` can serve any open-weights chat model and point **Claude Code** straight
at it — including GGUF-only models (via the `llama` engine) that Claude Code
could not otherwise use. One `gpu.yaml` is the whole configuration.

## Turnkey

```bash
cd my-project                 # holds gpu.yaml
gpu setup                     # provision the GPU, serve the model, start nothing else
gpu claude                    # launch Claude Code wired to the model
# ... work ...
gpu destroy                   # stop billing when done (or let it idle out)
```

`gpu claude` reads `gpu.yaml`, makes sure the gateway is up, and launches
`claude` with the Anthropic env it needs — base URL (the gateway), the model id
and aliases, and the context window (`runtime.context_length`, so Claude Code
does not assume a 1M window for an unknown model id). Arguments after
`gpu claude` are passed through to `claude`.

The `gpu.yaml` needs a gateway with the right adapter:

```yaml
gpu_runtime:
  model:
    id: peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF
  runtime:
    engine: llama            # GGUF via llama-server (OpenAI backend)
    quant: Q5_K_XL
    context_length: 262144   # what Claude Code will use as its window
  gateway:
    enabled: true
    adapter: anthropic-openai # translate Anthropic <-> OpenAI (see below)
    listen: 127.0.0.1:18090
```

- **`adapter: anthropic-openai`** — for an OpenAI-only backend (the `llama`
  engine, or vLLM's OpenAI routes). It translates Claude Code's Anthropic
  `/v1/messages` to `/v1/chat/completions` and back (streaming + tool calls).
- **`adapter: anthropic`** — only if the backend already serves `/v1/messages`
  natively (some vLLM builds). No translation.

For a persistent endpoint (survives reboot), `gpu gateway install` runs the
gateway as a systemd user service; then `gpu claude` just attaches to it.

## Project layout

`gpu init` scaffolds a reusable skeleton:

```
my-project/
├── gpu.yaml          # what to serve
├── CLAUDE.md         # how the agent should behave (read by Claude Code)
├── scripts/          # code the agent writes (helpers, one-off tools)
├── work/             # outputs and artifacts (results, data, reports)
└── .gpu/             # instance state (managed by gpu)
```

Keep code in `scripts/` and outputs in `work/`; tell the agent so in
`CLAUDE.md`. A smaller local model follows short, explicit rules best.

## Pre-approving MCP tools (operator guide)

Claude Code's **auto mode** asks the *model* to classify whether each tool call
is safe. A small local model is slow, so those extra classifier calls can time
out (`"model temporarily unavailable ... auto mode cannot determine the
safety"`) and block the work.

Fix it by **pre-approving** the tools the operator trusts, in the project's
`.claude/settings.json` — approved tools skip the classifier entirely:

```json
{
  "permissions": {
    "allow": [
      "mcp__kali__server_health",
      "mcp__kali__nmap_scan",
      "mcp__kali__nikto_scan",
      "mcp__kali__gobuster_scan"
    ]
  }
}
```

- Use the exact MCP tool names (`mcp__<server>__<tool>`), or `mcp__<server>` to
  allow a whole server.
- **Be deliberate with arbitrary-execution tools** (e.g. an `execute_command`):
  pre-approving one lets the agent run any command on that MCP's host without a
  prompt. Include it only in an environment you control (an authorized test
  box), and drop it when you want per-command review.

A larger model handles the deferred-tool protocol (`ToolSearch`) and parallel
tool calls well; with a smaller local model, also instruct it in `CLAUDE.md` to
call tools directly and run them one at a time.
