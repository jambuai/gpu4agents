---
name: gpu4agents
description: GPU for Agents. Rent a cloud GPU on your own, serve an open-weights model on it behind an OpenAI-compatible API, run experiments or evals, run its own code on the GPU as a job (training, fine-tuning, image/audio/video models, anything that needs CUDA) and bring the outputs back, monitor cost and idle time, and shut it down so the user pays only while it works. Works with Vast.ai out of the box and with any other GPU provider through an adapter you can write. Use when the user wants to run, test, benchmark or compare a HuggingFace model (Qwen, Llama, Mistral, ...) on a GPU, needs an LLM endpoint for an experiment, needs a GPU for code it writes (train a LoRA, generate images or audio, run a CUDA benchmark), asks what GPUs are running or what they cost, wants to use a specific GPU provider, or asks to install or set up GPU for Agents (gpu4agents, the `gpu` command).
---

# GPU for Agents

`gpu` (the `gpu4agents` package) rents a machine from a GPU provider, serves a model on it with vLLM behind an OpenAI-compatible API or runs your own code on it as a job, and installs a watchdog on the machine. The watchdog shuts the machine down once nothing has used it for `idle_timeout` (15 min by default), so billing ends even if this conversation does.

Vast.ai, RunPod and Lambda are built in. Any other provider (CoreWeave, Paperspace, a company cluster) plugs in through an adapter: one Python file that implements the provider contract. See "Another provider" below.

Each project directory has a `gpu.yaml` (what to rent and serve, and from which provider) and a `.gpu/` directory (what was rented). One project, one machine.

## Setup, once per machine

Do these in order. Skip a step whose check already passes.

### 1. Install the `gpu` command

Check: `gpu --version` prints 0.11.0 or later. First run `type gpu`: if it says `gpu` is an alias or a function (Oh My Zsh's git plugin makes it `git push upstream`), plain `gpu` runs that instead, so use `command gpu`, `~/.local/bin/gpu` or `gpu4agents` for every command in this session. In a git repository the alias would really push. If the command is missing or older, install it (Linux and macOS; on Windows, use WSL):

```bash
curl -fsSL https://gpu4agents.jambu.ai/install.sh | sh -s -- --agent <runtime>
```

`<runtime>` is the agent you are running in: `claude`, `codex`, `gemini`, `cursor`, or `auto`. If that address does not resolve, use `https://raw.githubusercontent.com/jambuai/gpu4agents/main/install.sh`. The script needs only `curl`. It installs [uv](https://docs.astral.sh/uv/) if missing, then runs `uv tool install --upgrade --python 3.12 gpu4agents`; uv downloads a suitable Python by itself, so the system Python does not matter. With `--agent` it also does step 3. The command lands in `~/.local/bin`. If this shell still does not find `gpu`, call `~/.local/bin/gpu` for the rest of the session.

To install by hand: `uv tool install --python 3.12 gpu4agents`, or `pipx install gpu4agents` with Python 3.10 or newer.

### 2. Connect a GPU provider account

Check: `gpu login --check` exits 0. It prints the provider, where the key was found, and the balance when the provider reports one.

If it exits 5, no account is connected yet. The user may not be technical, so skip jargon like "API key" and tell them, in these words or close:

> One step left, about two minutes: connect the account that pays for the GPUs. Open your terminal app (not this chat) and type `gpu login`. It shows the providers you can use and walks you through it: Vast.ai, RunPod and Lambda work right away, and if you already use another one (your company's servers, for example), pick "Another provider" and I'll add it. Your key stays on your computer; I never see it.

If they pick another provider, write its adapter ("Another provider" below), then ask them to run `gpu login --provider <name>` in their terminal.

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
| create gpu.yaml | `gpu_init(model)` | `gpu init --model ORG/NAME [--provider NAME]` |
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
| run your code on the GPU | `gpu_job_run(command, dir)` | `gpu job run -- CMD` |
| job state, log, cost | `gpu_job_status(job_id)` | `gpu job status`, `gpu job logs -f` |
| download job outputs | `gpu_job_pull(job_id)` | `gpu job pull` |
| stop a job | `gpu_job_cancel(job_id)` | `gpu job cancel` |
| everything billing in the account | `gpu_ls` | `gpu --json ls` |
| providers and what each can do | | `gpu --json providers` |

## Rules: this spends the user's money

1. Ask before renting. Before the first `gpu_setup` / `gpu setup` / `gpu run` in a conversation, show the model, the GPU, the price (`gpu_offers`) and the cap (`budget.max_hourly_cost_usd`), and get a yes. Never raise the budget in `gpu.yaml` without asking.
2. Setup is slow. The first setup takes 5 to 15 minutes (image pull, model download), and a bad host is replaced automatically. The MCP `gpu_setup` returns `status: "running"` while it works: call it again to keep waiting. From a shell, run `gpu --json setup` in the background with no timeout. Never start a second project, `--recreate`, or destroy because setup is slow. `gpu_status` (it shows `busy: true` during setup) and `gpu_logs(source="onstart")` show progress. If the user wants to give up, `gpu_stop` or `gpu_destroy` cancel the setup and destroy the half-booted machine.
3. Never handle the provider key. See step 2 of the setup. If a command fails with exit code 5, send the user to `gpu login`.
4. Clean up. When the work is done, offer `gpu_stop` (cheap to resume: about a minute, same key) or `gpu_destroy` (nothing billed; next time rents and downloads again). Destroy only with the user's explicit yes. Before finishing, run `gpu_ls` and report anything still running.

## A typical experiment

1. If there is no `gpu.yaml`: `gpu_init(model=..., max_hourly_cost_usd=...)`. Size the GPU for the model: about 2 GB of VRAM per billion parameters in bf16, plus room for the KV cache. A 7 to 9B model wants 24 GB; a 0.5 to 3B model fits on 16 to 24 GB. `gpu inspect-model ORG/NAME` suggests VRAM and a tool-call parser.
2. `gpu_validate`, then `gpu_offers`. Show the user the price and ask.
3. `gpu_setup` until `status: "ready"`. You get `base_url`, `api_key`, `model`.
4. Run the work:
   - a script: `gpu_run(command=["python", "eval.py"])` (or `gpu run python eval.py`). The script gets `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `JAMBU_MODEL_ID` in its environment, which any OpenAI SDK reads, and the watchdog keeps the machine up while it runs. Write the script to read those variables instead of pasting the key into code.
   - single requests: `curl "$base_url/chat/completions" -H "Authorization: Bearer $api_key"` with `"model": model`. Every request counts as activity.
5. Report results and cost (`gpu_status` has `session_cost_usd`), then rule 4.

## Your own code on the GPU (jobs)

When the work is code that needs a GPU, not requests to a model (training or fine-tuning, an image, audio or video model, a batch, a benchmark), run it as a job. The full contract is in https://gpu4agents.jambu.ai/jobs.md (copy in the repository: `JOBS.md`). In short:

1. A machine that only runs jobs: `gpu init --jobs --min-vram-gb N`. A project serving a model can run jobs too; the job reaches the model on `OPENAI_BASE_URL`.
2. Put the code in a directory with a `gpujob.yaml`: `run:` (the command), `setup:` (pinned installs), `env:`, `timeout:`, `exclude:`. `.git`, virtualenvs, `.env` files and `gpu-outputs/` never upload; the limit is 2 GB, so the job downloads large data itself.
3. The code writes every result to `$JAMBU_OUTPUT_DIR` (checkpoints too, as it goes), prints progress, exits non-zero on failure and never prints secrets. `HF_HOME` is a cache on the machine.
4. With the user's yes: `gpu job run` (uploads, runs, streams the log, downloads to `gpu-outputs/<job id>/`, exits with the job's code). Through MCP: `gpu_setup` until ready, `gpu_job_run`, `gpu_job_status` until the state is not `running`, `gpu_job_pull`.
5. A running job keeps the machine up; `max_lifetime` and the budget still stop it. Pull outputs before `gpu destroy`, which deletes them.

## Serve any model (images, voice, transcription, embeddings)

Any Hugging Face model can be served behind OpenAI routes, not only chat. The full contract is in https://gpu4agents.jambu.ai/serving.md (copy in the repository: `SERVING.md`). In short:

1. `gpu init --model <repo>` (MCP: `gpu_init`) reads the model card and writes the engine: `vllm` for chat, `hf` for the rest, with `model.task` one of `embeddings`, `image`, `transcription`, `speech`, `pipeline`. The output and `gpu_init`'s result name it; `--engine` and `--task` override.
2. `gpu setup` deploys it; `gpu env` (MCP: `gpu_connection`) gives `base_url`, `api_key`, `model` and `routes`. Call it with any OpenAI client: `images.generate`, `audio.speech.create`, `audio.transcriptions.create`, `embeddings.create`; other pipelines take `POST /v1/pipeline` with `{"inputs": ..., "parameters": {...}}`.
3. To switch models, edit `model.id` (and `model.task`) in gpu.yaml and run `gpu setup` again: same machine, new server. A larger `min_vram_gb` needs `--recreate`.
4. When the built-in server cannot load a model (its own library, several models chained, video), `gpu init --engine custom` writes `server/server.py` and `server/gpujob.yaml`. Fill in `load()` and the handlers; keep `/health` (503 until loaded), the `JAMBU_API_KEY` check and `/metrics` (`jambu_requests_running`, `jambu_requests_total`), and listen on `$PORT`.
5. `gpu logs` shows the server's log. Requests keep the machine up; it shuts down after `idle_timeout` once they stop (on Vast.ai it is destroyed, so the next setup downloads the model again).

## Another provider

When the user wants a provider `gpu providers` does not list, write an adapter for it. The full steps are in https://gpu4agents.jambu.ai/providers.md (copy in the repository: `PROVIDERS.md`). In short:

1. `gpu providers new <name>` writes `gpu_providers/<name>.py`, a skeleton of the provider contract. `gpu init --provider <name>` writes a `gpu.yaml` that loads it (`provider.name` and `provider.plugin`).
2. Read the provider's API docs and replace every `todo(...)` in that file with real code: offers, rent, status, stop, destroy, list, and the two stdlib-only calls the watchdog makes from the machine to stop or delete itself.
3. The user runs `gpu login --provider <name>`. Then `gpu validate` and `gpu offers` must work without renting anything.
4. One small test run, with the user's yes: `gpu setup`, `gpu status` (`watchdog.reachable` must be true), `gpu destroy --yes`.

Everything else in this skill (rules, commands, monitoring) works the same with any provider.

## Monitoring

- `gpu --json status` (or `gpu_status`): `state`, `hourly_cost_usd`, `session_cost_usd`, `idle_seconds`, `locked_until`, `decision` (what the watchdog will do next and why), `watchdog.reachable`, `workloads`, and `busy` while a setup runs. Report cost and idle time when the user asks how things are going, and before a long wait.
- `gpu --json logs --source runtime|onstart|watchdog|provider`: the model server, the boot script, the watchdog, or the container log from the provider.
- `gpu events`: the timeline of this project (rented, ready, idle, stopped, and why).
- `gpu --json ls` (or `gpu_ls`): every machine billing in the account, with the project that owns it or `orphan`.
- `gpu sweep` shows what would be stopped or destroyed account-wide, and `gpu sweep --apply` does it. `gpu sweep --install-timer` (systemd on Linux, launchd on macOS) runs it every few minutes as a safety net. Suggest the timer once to a user who rents often.

## When something fails

Exit codes are the same for the CLI and the MCP tools (`exit_code` in the error):

| Code | Meaning | Do |
| --- | --- | --- |
| 2 | config error, or gpu.yaml changed since setup | fix it; `gpu_setup(recreate=true)` replaces a machine built from the old config |
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
- `gpu` prints a git error (`fatal: not a git repository`, or it pushed): `gpu` is a shell alias (Oh My Zsh). Use `command gpu` or `gpu4agents`, and tell the user the installer can remove the alias for new terminals.
- Requests fail with connection refused or a timeout: the machine was shut down for being idle. Call `gpu_setup` again (about a minute after a stop, a few minutes after a destroy), take the new `base_url` from its output, and retry once.
- `401`: a new machine was rented and the key changed; take it from `gpu_connection`.
- A long job with quiet stretches longer than `idle_timeout`: `gpu_lock(duration="2h")`, then `gpu_unlock`.

Docs: https://github.com/jambuai/gpu4agents/tree/main/docs · Website: https://gpu4agents.jambu.ai
