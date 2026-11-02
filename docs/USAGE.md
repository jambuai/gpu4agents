# Using gpu

Operator reference. The project front page is the [README](../README.md). Field-by-field config is [CONFIG.md](CONFIG.md).

## Commands

| Command | What it does |
| --- | --- |
| `gpu init` | Write a starter `jambu.yaml` and `.env.example` (`--flat` for a legacy `config.yml`) |
| `gpu inspect-model <repo>` | Suggest `min_vram_gb`, `dtype`, and `tool_calling` for a HuggingFace model, with sources cited. `--add-profile <name>` writes it into the model catalog |
| `gpu profiles` | List the model profiles in `jambu.models.yaml` |
| `gpu providers` | Provider capability matrix and credential status |
| `gpu validate` | Schema, policy, credentials, capabilities, real capacity. Fails before provisioning |
| `gpu offers` | Matching GPU offers and their hourly price |
| `gpu setup` | Provision, install the watchdog, start the runtime, wait for READY |
| `gpu run <cmd>` | Guarantee READY, then run the command with heartbeats |
| `gpu status` | Reconcile local and provider state, show the guard decision |
| `gpu lock --for 2h` | Prevent automatic shutdown until the TTL expires |
| `gpu unlock` | Resume normal lifecycle rules |
| `gpu stop` | Stop compute. Idempotent |
| `gpu destroy` | Destroy the remote resource. Idempotent |
| `gpu logs` | The model server's log on the instance (`--source runtime\|onstart\|watchdog\|provider`, `--lines N`). Falls back to the provider's container log when the watchdog does not answer |
| `gpu events` | Structured lifecycle event log |
| `gpu guard` | Evaluate and enforce the lifecycle rule from cron or CI |
| `gpu ls` | Every jambu instance in the provider account, and which registered project owns it |
| `gpu sweep` | Guard every registered project by its own rules, then report (or `--apply`) stops and destroys of orphans |
| `gpu sweep --install-timer` | Write a systemd user timer running `sweep --apply`, and register this project |
| `gpu endpoint` | Print the model endpoint URL |
| `gpu env` | Print the base URL, API key and model as shell exports (`--format dotenv\|json`), without starting anything |
| `gpu agent install` | Give Claude Code (`-t claude`), Cursor (`-t cursor`) or Claude Desktop (`-t desktop`) the gpu skill and MCP server ([AGENTS.md](AGENTS.md)) |
| `gpu agent skill` | Print the bundled `SKILL.md` |
| `gpu mcp` | The local MCP server on stdio, started by the agent's client |

Every command accepts `--json` and `--config PATH`. To call `gpu` from an application in any language, see [INTEGRATION.md](INTEGRATION.md).

## The manifest

`gpu` reads only the `gpu_runtime:` key. Everything else in `jambu.yaml` belongs to another tool. Credentials never go in the file.

A project that does not share the file can run `gpu init --flat`. That writes a `config.yml` with the same fields at the top level, without the `gpu_runtime:` wrapper. When both files exist, `jambu.yaml` wins.

More than one model: do not copy the whole file per model. Set `model_profile: <name>` and keep the definitions in [`jambu.models.yaml`](../jambu.models.yaml). `gpu inspect-model <repo> --add-profile <name>` writes a profile. `gpu profiles` lists them. See [CHOOSING_A_MODEL.md](CHOOSING_A_MODEL.md).

The full field list is [CONFIG.md](CONFIG.md). A filled example is [`jambu.example.yaml`](../jambu.example.yaml).

## Running a workload

`gpu run` executes the command locally, against the remote model, and injects:

| Variable | Value |
| --- | --- |
| `JAMBU_GPU_ENDPOINT` | `http://<ip>:<port>` |
| `OPENAI_BASE_URL` | `http://<ip>:<port>/v1` |
| `OPENAI_API_KEY` | the endpoint's key: `runtime.api_key`, otherwise the one generated for this instance |
| `JAMBU_MODEL_ID` | the name to request: `model.served_name`, otherwise `model.id` |
| `JAMBU_INSTANCE_ID`, `JAMBU_WORKLOAD_ID`, `JAMBU_PROVIDER` | identity |
| `JAMBU_SSH_HOST`, `JAMBU_SSH_PORT` | when the provider exposes SSH |

```python
import os
from openai import OpenAI

client = OpenAI()
print(client.chat.completions.create(
    model=os.environ["JAMBU_MODEL_ID"],
    messages=[{"role": "user", "content": "hi"}],
).choices[0].message.content)
```

Set `workload.mode: remote` to run the command on the instance over SSH.

The instance is not stopped after each command. `idle_timeout` governs reuse, so the next experiment does not provision again.

## Agents

vLLM already serves the agent APIs, so gpu adds no orchestration layer. It turns on the right vLLM flags and gives the workload the same endpoint.

- `/v1/chat/completions` with `tools=[...]`: the OpenAI tool-calling shape used by LangChain, LangGraph, CrewAI, and others.
- `/v1/responses`: the OpenAI Responses API, including MCP tool routing done by vLLM when `runtime.tool_server` is set.

```yaml
runtime:
  engine: vllm
  tool_calling: auto
  # tool_call_parser: hermes
  # tool_server: https://your-mcp-server
```

`tool_calling: auto` guesses a parser from `model.id`. Set `tool_call_parser` when vLLM cannot guess. The list is in [CONFIG.md](CONFIG.md).

[`examples/mastra-agent/`](../examples/mastra-agent/) is a Mastra agent that uses the injected environment variables:

```bash
gpu setup
gpu run npm --prefix examples/mastra-agent start
```

`gpu status` shows the active parser, for example `vLLM :8000 (model, tools=hermes)`. Tool calls fire only after a parser is on.

## Keeping the account clean

The watchdog guards an instance from the inside, and `gpu guard` guards the one instance a project tracks. Neither helps when the watchdog died and the project directory is gone. `gpu ls` and `gpu sweep` look at the whole provider account instead. They only ever see instances whose label starts with `jambu-`, the label `gpu setup` gives every instance. Nothing else in the account is listed or touched.

```bash
gpu ls
```

```
Vast.ai instances
 ID        LABEL                          STATE    GPU          $/H     AGE    PROJECT
 31337001  jambu-Qwythos-9B-v2-3fa2c1     READY    1x RTX 4090  $0.350  7h30m  /home/me/agent/jambu.yaml
 31337002  jambu-Llama-3.1-8B-9b01de      STOPPED  1x RTX 4090  $0.350  1d6h   orphan
```

### Every project by its own rules

`gpu setup` (and `gpu run`, which can provision) registers the project's absolute config path in `~/.gpu/projects.json` (`$GPU_HOME` overrides `~/.gpu`; a `~/.jambu` from before 0.3.0 is renamed to `~/.gpu` on first use, and `$JAMBU_HOME` is still read). `gpu sweep` then works in two steps:

1. Tracked instances. For every registered project whose config still loads, it runs that project's own `gpu guard`: its own `jambu.yaml` and `.gpu/state.json`, so its own `max_lifetime`, lock, session start, `stopped_ttl` and budget. The project's events land in its own log. A registered project whose config file is gone is dropped from the registry. The project in the current directory, or the one given with `--config`, is guarded too for that run.
2. Orphans. An instance no project tracks has no known policy, and nobody knows whether it is in use. The rules for it are simple, and they never destroy a running machine:

| Orphan | Rule | Action |
| --- | --- | --- |
| running | running longer than `--orphan-max-age` (default `24h`, from the provider's creation time) | `stop`, never `destroy` |
| stopped | first seen stopped by the sweep more than `--stopped-ttl` ago (default `24h`) | `destroy` |

An instance whose project exists but could not be guarded this run (the config no longer parses, another `gpu` command holds the lock, an API error) still counts as tracked: the sweep reports the error and exits 1, but never treats that instance as an orphan.

It is a dry run until you pass `--apply`:

```
$ gpu sweep
> guarding 1 project(s) (dry run)
= /home/me/agent/jambu.yaml: 31337001 ok (within_policy)
> sweeping Vast.ai for orphans: running > 1d is stopped, stopped > 1d is destroyed (dry run)
- would destroy orphan 31337002 (jambu-Llama-3.1-8B-9b01de): stopped_ttl, stopped for 1d2h
! 1 action(s) pending; re-run with --apply
```

Providers do not report when an instance stopped, so the sweep keeps the time it first saw each orphan stopped in `~/.gpu/sweep.json`. An entry goes away when the instance runs again, disappears, or is tracked by a project. `stopped_ttl` for an orphan therefore counts from the first sweep that saw it stopped, never from a project's state. Every orphan action is recorded as a `sweep_action` event with its reason in `~/.gpu/logs/events.jsonl`. Credentials for the orphan step resolve the usual way (the environment, `.env` in the working directory, `~/.gpu/credentials`), then from the `.env` of registered projects.

### Run it on a timer

```bash
gpu sweep --install-timer            # every 5m; --every 15m to change
systemctl --user daemon-reload && systemctl --user enable --now jambu-gpu-sweep.timer
```

`--install-timer` writes `~/.config/systemd/user/jambu-gpu-sweep.service` and `.timer`, and registers the current project if there is one. The service is a single `ExecStart=<python> -m gpu4agents sweep --apply` with the interpreter that installed it, run from `~`. The registry decides which projects get guarded, and the working directory plays no part, so installing from a second project changes nothing in the unit. The oneshot has a 10 minute start timeout, a failing sweep shows as a failed unit, and `GPU_HOME` is passed through when it is set. A unit written by an older version, with one `guard` line per project, is rewritten and those projects are registered.

A user unit does not inherit your shell's environment. `--install-timer` therefore refuses unless the provider key is in a file: the project `.env` or `~/.gpu/credentials`. A key that is only exported in your shell is not enough. A user timer needs no root, but it stops at logout unless lingering is on (`loginctl enable-linger $USER`).

```bash
gpu sweep --uninstall-timer
systemctl --user disable --now jambu-gpu-sweep.timer
```
