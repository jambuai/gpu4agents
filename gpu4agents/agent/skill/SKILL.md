---
name: gpu4agents
description: GPU for Agents. Rent a cloud GPU on your own, serve an open-weights model on it behind an OpenAI-compatible API, run experiments or evals, monitor cost and idle time, and shut it down so the user pays only while it works. Works with Vast.ai out of the box and with any other GPU provider through an adapter you can write. Use when the user wants to run, test, benchmark or compare a HuggingFace model (Qwen, Llama, Mistral, ...) on a GPU, needs an LLM endpoint for an experiment, asks what GPUs are running or what they cost, wants to use a specific GPU provider, or asks to install or set up GPU for Agents (gpu4agents, the `gpu` command).
---

# GPU for Agents

`gpu` (the `gpu4agents` package) rents a machine from a GPU provider, serves a model on it with vLLM behind an OpenAI-compatible API, and installs a watchdog on the machine. The watchdog shuts the machine down once nothing has used it for `idle_timeout` (15 min by default), so billing ends even if this conversation does.

Vast.ai is built in. Any other provider (RunPod, Lambda, a company cluster) plugs in through an adapter: one Python file that implements the provider contract. See "Another provider" below.

Each project directory has a `jambu.yaml` (what to rent and serve, and from which provider) and a `.gpu/` directory (what was rented). One project, one machine.

## Setup, once per machine

Do these in order. Skip a step whose check already passes.

### 1. Install the `gpu` command

Check: `gpu --version` prints 0.7.0 or later. If the command is missing or older, install it (Linux and macOS; on Windows, use WSL):

```bash
curl -fsSL https://gpu4agents.jambu.ai/install.sh | sh -s -- --agent <runtime>
```

`<runtime>` is the agent you are running in: `claude`, `codex`, `gemini`, `cursor`, or `auto`. If that address does not resolve, use `https://raw.githubusercontent.com/jambuai/gpu4agents/main/install.sh`. The script needs only `curl`. It installs [uv](https://docs.astral.sh/uv/) if missing, then runs `uv tool install --upgrade --python 3.12 gpu4agents`; uv downloads a suitable Python by itself, so the system Python does not matter. With `--agent` it also does step 3. The command lands in `~/.local/bin`. If this shell still does not find `gpu`, call `~/.local/bin/gpu` for the rest of the session.

To install by hand: `uv tool install --python 3.12 gpu4agents`, or `pipx install gpu4agents` with Python 3.10 or newer.

### 2. Connect a GPU provider account

Check: `gpu login --check` exits 0. It prints the provider, where the key was found, and the balance when the provider reports one.

If it exits 5, there is no working key yet. Ask the user which provider they want, then tell them, in these words or close:

- Vast.ai (built in): "Create an API key at https://cloud.vast.ai/manage-keys/ and add a few dollars of credit. Then open your own terminal, run `gpu login`, and paste the key into the hidden prompt."
- Any other provider: write its adapter first ("Another provider" below), then ask the user to run `gpu login --provider <name>` in their own terminal.

Wait for them to say it is done, then run `gpu login --check` again.

Never ask the user to paste a key into this conversation, never read `~/.gpu/credentials` or `.env`, and never print a key. The key stays between the user's terminal and the provider.

### 3. Register the gpu tools in this agent (optional)

```bash
gpu agent install -t claude --global     # Claude Code
gpu agent install -t codex --global      # Codex
gpu agent install -t gemini --global     # Gemini CLI
gpu agent install -t cursor --global     # Cursor
```

This writes this skill and registers the `gpu` MCP server for that agent. The tools appear after the user restarts the agent. Until then, and in any agent without MCP, the CLI below does everything.

## Two ways to drive it

- MCP tools (`gpu_*`), when the `gpu` MCP server is connected. Prefer them.
- The CLI, from a shell: `gpu --json <command>` prints one JSON document on stdout. Errors go to stderr as `error: ...` with a non-zero exit code. Run it from the project directory.

| Step | MCP tool | CLI |
| --- | --- | --- |
| create jambu.yaml | `gpu_init(model)` | `gpu init --model ORG/NAME [--provider NAME]` |
| check config and credentials | `gpu_validate` | `gpu --json validate` |
| see prices, rent nothing | `gpu_offers` | `gpu --json offers` |
| start / ensure ready | `gpu_setup` | `gpu --json setup` |
| endpoint without starting | `gpu_connection` | `gpu --json env` |
| run a job against it | `gpu_run(command=[...])` | `gpu run python job.py` |
| state, cost, idle time | `gpu_status` | `gpu --json status` |
| boot / server / watchdog logs | `gpu_logs(source)` | `gpu --json logs --source onstart` |
| keep up through a quiet stretch | `gpu_lock` / `gpu_unlock` | `gpu lock --for 2h` / `gpu unlock` |
| stop (keep disk) | `gpu_stop` | `gpu --json stop` |
| delete | `gpu_destroy(confirm=true)` | `gpu --json destroy --yes` |
| everything billing in the account | `gpu_ls` | `gpu --json ls` |
| providers and what each can do | | `gpu --json providers` |

## Rules: this spends the user's money

1. Ask before renting. Before the first `gpu_setup` / `gpu setup` / `gpu run` in a conversation, show the model, the GPU, the price (`gpu_offers`) and the cap (`budget.max_hourly_cost_usd`), and get a yes. Never raise the budget in `jambu.yaml` without asking.
2. Setup is slow. The first setup takes 5 to 15 minutes (image pull, model download), and a bad host is replaced automatically. The MCP `gpu_setup` returns `status: "running"` while it works: call it again to keep waiting. From a shell, run `gpu --json setup` in the background with no timeout. Never start a second project, `--recreate`, or destroy because setup is slow. `gpu_status` (it shows `busy: true` during setup) and `gpu_logs(source="onstart")` show progress. If the user wants to give up, `gpu_stop` or `gpu_destroy` cancel the setup and destroy the half-booted machine.
3. Never handle the provider key. See step 2 of the setup. If a command fails with exit code 5, send the user to `gpu login`.
4. Clean up. When the work is done, offer `gpu_stop` (cheap to resume: about a minute, same key) or `gpu_destroy` (nothing billed; next time rents and downloads again). Destroy only with the user's explicit yes. Before finishing, run `gpu_ls` and report anything still running.

## A typical experiment

1. If there is no `jambu.yaml`: `gpu_init(model=..., max_hourly_cost_usd=...)`. Size the GPU for the model: about 2 GB of VRAM per billion parameters in bf16, plus room for the KV cache. A 7 to 9B model wants 24 GB; a 0.5 to 3B model fits on 16 to 24 GB. `gpu inspect-model ORG/NAME` suggests VRAM and a tool-call parser.
2. `gpu_validate`, then `gpu_offers`. Show the user the price and ask.
3. `gpu_setup` until `status: "ready"`. You get `base_url`, `api_key`, `model`.
4. Run the work:
   - a script: `gpu_run(command=["python", "eval.py"])` (or `gpu run python eval.py`). The script gets `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `JAMBU_MODEL_ID` in its environment, which any OpenAI SDK reads, and the watchdog keeps the machine up while it runs. Write the script to read those variables instead of pasting the key into code.
   - single requests: `curl "$base_url/chat/completions" -H "Authorization: Bearer $api_key"` with `"model": model`. Every request counts as activity.
5. Report results and cost (`gpu_status` has `session_cost_usd`), then rule 4.

## Another provider

When the user wants a provider `gpu providers` does not list, write an adapter for it. The full steps are in https://gpu4agents.jambu.ai/providers.md (copy in the repository: `PROVIDERS.md`). In short:

1. `gpu providers new <name>` writes `gpu_providers/<name>.py`, a skeleton of the provider contract. `gpu init --provider <name>` writes a `jambu.yaml` that loads it (`provider.name` and `provider.plugin`).
2. Read the provider's API docs and fill in every method that raises `NotImplementedError`: offers, rent, status, stop, destroy, list, and the two stdlib-only calls the watchdog makes from the machine to stop or delete itself.
3. The user runs `gpu login --provider <name>`. Then `gpu validate` and `gpu offers` must work without renting anything.
4. One small test run, with the user's yes: `gpu setup`, `gpu status` (`watchdog.reachable` must be true), `gpu destroy --yes`.

Everything else in this skill (rules, commands, monitoring) works the same with any provider.

## Monitoring

- `gpu --json status` (or `gpu_status`): `state`, `hourly_cost_usd`, `session_cost_usd`, `idle_seconds`, `locked_until`, `decision` (what the watchdog will do next and why), `watchdog.reachable`, `workloads`, and `busy` while a setup runs. Report cost and idle time when the user asks how things are going, and before a long wait.
- `gpu --json logs --source runtime|onstart|watchdog|provider`: the model server, the boot script, the watchdog, or the container log from the provider.
- `gpu events`: the timeline of this project (rented, ready, idle, stopped, and why).
- `gpu --json ls` (or `gpu_ls`): every machine billing in the account, with the project that owns it or `orphan`.
- `gpu sweep` shows what would be stopped or destroyed account-wide, and `gpu sweep --apply` does it. `gpu sweep --install-timer` (Linux, systemd) runs it every few minutes as a safety net. Suggest the timer once to a user who rents often.

## When something fails

Exit codes are the same for the CLI and the MCP tools (`exit_code` in the error):

| Code | Meaning | Do |
| --- | --- | --- |
| 2 | config error, or jambu.yaml changed since setup | fix it; `gpu_setup(recreate=true)` replaces a machine built from the old config |
| 3 | no machine / no endpoint | `gpu_setup` |
| 4, 8 | provider API error or timeout | retry with backoff |
| 5 | provider auth failed, or no key | the user runs `gpu login` in a terminal (setup step 2) |
| 6 | no capacity matches | retry later, or relax `compute.gpu` / the budget (ask first) |
| 7 | provisioning failed | retry; the failed machine is already destroyed |
| 9 | the provider cannot do this | change the config, or the adapter's capabilities |
| 10 | over budget | ask the user before raising it |
| 11 | model server did not start | the error carries the vLLM log tail: usually too little VRAM, a gated model without `HF_TOKEN`, or a bad `runtime` option. The machine was destroyed |
| 12 | another gpu command (a setup) holds the project | follow it with `gpu_setup`; `gpu_stop` / `gpu_destroy` cancel it |

- `gpu: command not found` right after installing: use `~/.local/bin/gpu`, or open a new shell.
- Requests fail with connection refused or a timeout: the machine was stopped for being idle. Call `gpu_setup` again (about a minute), take the new `base_url` from its output, and retry once.
- `401`: a new machine was rented and the key changed; take it from `gpu_connection`.
- A long job with quiet stretches longer than `idle_timeout`: `gpu_lock(duration="2h")`, then `gpu_unlock`.

Docs: https://github.com/jambuai/gpu4agents/tree/main/docs · Website: https://gpu4agents.jambu.ai
