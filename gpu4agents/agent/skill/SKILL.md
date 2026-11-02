---
name: gpu
description: Rent a temporary cloud GPU, serve an open-weights model on it behind an OpenAI-compatible API, run experiments or evals against it, and shut it down so it stops billing. Use when the user wants to run, test, benchmark or compare a HuggingFace model (Qwen, Llama, Mistral, ...) on a GPU, needs an LLM endpoint for an experiment, or asks about GPU instances gpu/gpu4agents created. Works through the `gpu` MCP tools (gpu_setup, gpu_run, ...) or the `gpu` CLI.
---

# gpu: temporary GPUs for experiments

`gpu` (the `gpu4agents` package) rents a machine on Vast.ai, serves a model with vLLM behind an OpenAI-compatible API, and installs a watchdog on the machine that stops it once nothing has used it for `idle_timeout` (15 min by default). A machine you forget stops by itself, so billing ends even if this conversation does.

Each project directory has a `jambu.yaml` (what to rent and serve) and a `.gpu/` directory (what was rented). One project, one machine.

## Two ways to drive it

- **MCP tools** (`gpu_*`), when the `gpu` MCP server is connected. Prefer them.
- **CLI**, from a shell: `gpu --json <command>` prints one JSON document on stdout; errors go to stderr as `error: ...` with a non-zero exit code. Run it from the project directory.

| Step | MCP tool | CLI |
| --- | --- | --- |
| create jambu.yaml | `gpu_init(model)` | `gpu init --model ORG/NAME` |
| check config and credentials | `gpu_validate` | `gpu --json validate` |
| see prices, rent nothing | `gpu_offers` | `gpu --json offers` |
| start / ensure ready | `gpu_setup` | `gpu --json setup` |
| endpoint without starting | `gpu_connection` | `gpu --json env` |
| run a job against it | `gpu_run(command=[...])` | `gpu run python job.py` |
| state, cost, idle time | `gpu_status` | `gpu --json status` |
| boot / server logs | `gpu_logs(source)` | `gpu --json logs --source onstart` |
| stop (keep disk) | `gpu_stop` | `gpu --json stop` |
| delete | `gpu_destroy(confirm=true)` | `gpu --json destroy --yes` |
| everything billing in the account | `gpu_ls` | `gpu --json ls` |

## Rules: this spends the user's money

1. **Ask before renting.** Before the first `gpu_setup` / `gpu setup` / `gpu run` in a conversation, show the model, the GPU and the price (`gpu_offers`) and the cap (`budget.max_hourly_cost_usd`), and get a yes. Never raise the budget in `jambu.yaml` without asking.
2. **Setup is slow, and that is normal.** The first setup takes 5–15 minutes (image pull, model download); a bad host is replaced automatically. The MCP `gpu_setup` returns `status: "running"` while it works: call it again to keep waiting. From a shell, run `gpu --json setup` in the background with no timeout. Never start a second project, `--recreate`, or destroy because setup is slow. `gpu_status` (it shows `busy: true` during setup) and `gpu_logs(source="onstart")` show progress. If the user wants to give up, `gpu_stop` or `gpu_destroy` cancel the setup and destroy the half-booted machine.
3. **Never handle the provider key.** Do not print, echo or ask for `VAST_API_KEY`. It is read from the environment, the project `.env` or `~/.gpu/credentials`. If it is missing (exit code 5), tell the user where to put it.
4. **Clean up.** When the work is done, offer `gpu_stop` (cheap to resume: about a minute, same key) or `gpu_destroy` (nothing billed; next time rents and downloads again). Destroy only with the user's explicit yes. Before finishing, run `gpu_ls` and report anything still running.

## A typical experiment

1. If there is no `jambu.yaml`: `gpu_init(model=..., max_hourly_cost_usd=...)`. Size the GPU for the model: about 2 GB of VRAM per billion parameters in bf16, plus room for the KV cache (a 7–9B model wants 24 GB; a 0.5–3B model fits on 16–24 GB). `gpu inspect-model ORG/NAME` suggests VRAM and a tool-call parser.
2. `gpu_validate`, then `gpu_offers`. Show the user the price; ask.
3. `gpu_setup` until `status: "ready"`. You get `base_url`, `api_key`, `model`.
4. Run the work:
   - a script: `gpu_run(command=["python", "eval.py"])` (or `gpu run python eval.py`). It gets `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `JAMBU_MODEL_ID` in its environment, which any OpenAI SDK reads, and the watchdog keeps the machine up while it runs. Write the script to read those variables instead of pasting the key into code.
   - single requests: `curl "$base_url/chat/completions" -H "Authorization: Bearer $api_key"` with `"model": model`. Every request counts as activity.
5. Report results and cost (`gpu_status` has `session_cost_usd`), then step 4 of the rules.

## When something fails

Exit codes are the same for the CLI and the MCP tools (`exit_code` in the error):

| Code | Meaning | Do |
| --- | --- | --- |
| 2 | config error, or jambu.yaml changed since setup | fix it; `gpu_setup(recreate=true)` replaces a machine built from the old config |
| 3 | no machine / no endpoint | `gpu_setup` |
| 4, 8 | provider API error or timeout | retry with backoff |
| 5 | provider auth failed | the user fixes `VAST_API_KEY` |
| 6 | no capacity matches | retry later, or relax `compute.gpu` / the budget (ask first) |
| 7 | provisioning failed | retry; the failed machine is already destroyed |
| 10 | over budget | ask the user before raising it |
| 11 | model server did not start | the error carries the vLLM log tail: usually too little VRAM, a gated model without `HF_TOKEN`, or a bad `runtime` option. The machine was destroyed |
| 12 | another gpu command (a setup) holds the project | follow it with `gpu_setup`; `gpu_stop` / `gpu_destroy` cancel it |

- Requests fail with connection refused or a timeout: the machine was stopped for being idle. Call `gpu_setup` again (about a minute), take the **new** `base_url` from its output, and retry once.
- `401`: a new machine was rented and the key changed; take it from `gpu_connection`.
- A long job with quiet stretches longer than `idle_timeout`: `gpu_lock(duration="2h")`, then `gpu_unlock`.

Full contract: https://github.com/jambuai/gpu4agents/blob/main/docs/INTEGRATION.md
