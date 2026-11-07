# Letting a coding agent rent GPUs

`gpu` ships a skill and a local MCP server, so Claude Code, Codex, Gemini CLI, Cursor or Claude Desktop can rent a GPU, serve a model, run an experiment against it and shut it down, from a request like "compare Qwen 7B and Llama 8B on these 50 prompts".

A watchdog on the machine makes this safe to hand to an agent. A machine the agent forgets, or a conversation that ends mid-experiment, shuts down by itself after `lifecycle.idle_timeout`.

## Install: let the agent do it

Paste into the agent, naming it:

```text
Set up GPU for Agents for me. I'm using Codex. Fetch https://gpu4agents.jambu.ai/install.md with curl and follow its steps.
```

[`INSTALL.md`](../INSTALL.md) (served at `/install.md`) tells the agent to run [`install.sh`](../install.sh) with `--agent <runtime>`, check the account with `gpu login --check`, and send the user to `gpu login` if there is no key. If the user wants a provider other than Vast.ai, the agent offers to write its adapter with [`PROVIDERS.md`](../PROVIDERS.md). Last, it reads the skill with `gpu agent skill`. Nothing is rented during installation.

`install.sh` needs only `curl` on Linux or macOS (Windows: WSL). It installs [uv](https://docs.astral.sh/uv/) if missing and runs `uv tool install --upgrade --python 3.12 gpu4agents`, so uv downloads a Python when the machine has no suitable one. It removes an old `jambu-gpu` tool, then runs `gpu agent install -t <runtime> --global` (`auto` by default: every runtime it finds). Options: `--agent claude|codex|gemini|cursor|auto|none`, `--no-modify-path`; `GPU4AGENTS_SPEC` installs a pinned version or a wheel.

## Install the skill only

The skill's first section installs the `gpu` command when it is missing, so the skill alone is enough to start. It lives in [`skills/gpu4agents/`](../skills/gpu4agents/SKILL.md).

| Agent | Command |
| --- | --- |
| Claude Code | `/plugin marketplace add jambuai/gpu4agents`, then `/plugin install gpu4agents@gpu4agents` |
| Codex | `npx skills add jambuai/gpu4agents --global --agent codex` |
| Gemini CLI, Cursor, others | `npx skills add jambuai/gpu4agents --global --agent '*'` |
| Claude.ai, Claude Desktop | upload [`gpu4agents-skill.zip`](https://github.com/jambuai/gpu4agents/releases/latest/download/gpu4agents-skill.zip) under Settings > Capabilities > Skills |

The repository is a Claude Code plugin marketplace (`.claude-plugin/`) and a Cursor plugin (`.cursor-plugin/`). `skills/gpu4agents/agents/openai.yaml` holds the display name and default prompt for Codex.

## Connect the account: `gpu login`

The user, in their own terminal:

```bash
gpu login                     # the first time: pick a provider, follow the steps, paste the key (hidden)
gpu login --provider runpod   # straight to one provider (built in, or one added with an adapter)
gpu login --check             # exit 0, the provider and where the key was found; or exit 5
echo "$KEY" | gpu login --stdin
```

Run with no project and no saved provider, `gpu login` first lists the providers it can use: Vast.ai, RunPod and Lambda, ready out of the box, and "another provider I already use", which tells the user to ask their agent to add it ([`PROVIDERS.md`](../PROVIDERS.md)). For the chosen provider it then shows the steps in plain words (create the account, add credit or a card, copy the key) before the hidden prompt. An adapter supplies its own steps with `tagline`, `signup_url` and `setup_steps`.

Otherwise the provider is the one `--provider` names, else the project's `gpu.yaml`, else `GPU_PROVIDER`, else Vast.ai. `gpu login --provider NAME` also saves `GPU_PROVIDER=NAME`, so `gpu ls` and `gpu sweep` outside a project use that provider, and `gpu init` writes it into a new `gpu.yaml`. The key goes to `~/.gpu/credentials` (mode 600, directory 700); other lines in the file are kept. A key in the environment or a project `.env` still works and wins over the file. The skill tells the agent never to ask for the key in chat, read those files, or print it.

## When `gpu` is a shell alias

Oh My Zsh's git plugin, on by default, defines `gpu` as `git push upstream`. An alias wins over the command, in a terminal and in the agent's shell, so `gpu login` runs git instead (and in a repository with an `upstream` remote, it pushes). The installer looks for a `gpu` alias in zsh and bash and adds a marked block at the end of `~/.zshrc` or `~/.bashrc` that removes it for new shells; `--keep-gpu-alias` leaves it. The same command is always installed as `gpu4agents` too, and the skill tells the agent to run `type gpu` first and use `command gpu` when it is an alias. The MCP server is unaffected: clients start it without a shell.

## Install by hand

```bash
uv tool install --python 3.12 gpu4agents    # or: pipx install gpu4agents
gpu agent install -t claude                 # this project; add --global for every project
gpu agent install -t auto --global          # every runtime on this machine
```

| Target | Skill | MCP server |
| --- | --- | --- |
| `claude` | `.claude/skills/gpu4agents/SKILL.md` | `.mcp.json` (Claude Code asks once to approve it); `--global`: `claude mcp add --scope user` |
| `codex` | `.agents/skills/gpu4agents/SKILL.md` | `[mcp_servers.gpu]` in `$CODEX_HOME/config.toml` (`~/.codex`), always user-wide |
| `gemini` | `.agents/skills/gpu4agents/SKILL.md` | `mcpServers.gpu` in `.gemini/settings.json` |
| `cursor` | `.agents/skills/gpu4agents/SKILL.md` | `.cursor/mcp.json`, with `--project ${workspaceFolder}` |
| `desktop` | `~/.gpu/gpu-skill.zip`, to upload under Settings > Capabilities > Skills | `claude_desktop_config.json`, pinned to this project |
| `auto` | every runtime found: its command on `PATH` or its config directory | |

`.agents/skills/` is the shared directory Codex, Gemini CLI and Cursor all read. `--global` puts skills and settings under the home directory instead of the project. Existing config files are merged; only the `gpu` entry is written, and a skill installed by an older version under the name `gpu` is replaced. Re-running is safe. Restart the agent afterwards so it loads the skill and the tools.

Claude Desktop and Cursor do not get your shell's environment. Keep the provider key in `~/.gpu/credentials` (`gpu login`) or the project `.env`, not only in an `export`.

Another agent that reads `SKILL.md` files: `gpu agent skill > path/to/skills/gpu4agents/SKILL.md`. Any MCP client: run `gpu mcp --project /path/to/project` over stdio.

## The MCP tools

| Tool | Does | Costs money |
| --- | --- | --- |
| `gpu_init` | write `gpu.yaml` for a model, with an optional price cap and VRAM | no |
| `gpu_validate` | check the config and credentials | no |
| `gpu_offers` | cheapest matching machines and their price | no |
| `gpu_setup` | rent / restart and wait for the model; returns `base_url`, `api_key`, `model` | yes |
| `gpu_status` | state, cost so far, idle time, the watchdog's next move, setup progress | no |
| `gpu_connection` | `base_url`, `api_key`, `model` without starting anything | no |
| `gpu_run` | run a local command with `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `JAMBU_MODEL_ID` set | yes, if it has to start the machine |
| `gpu_logs` | runtime, onstart, watchdog or provider log | no |
| `gpu_lock` / `gpu_unlock` | keep the machine up through a quiet period | while locked |
| `gpu_stop` | stop now; disk kept, restart takes about a minute | ends billing for compute |
| `gpu_destroy` | delete; needs `confirm=true`; downloads job results first, `discard_outputs=true` deletes them | ends all billing |
| `gpu_ls` | every machine in the account, from any project | no |

Each tool runs the `gpu` CLI with `--json` in the project directory, so errors carry the same `exit_code` as the [application contract](INTEGRATION.md#exit-codes), plus `meaning` and `next` for the agent. A tool takes an optional `project_dir`; by default it is the server's `--project`, `$GPU_PROJECT_DIR`, or the directory the client started it in.

`gpu_setup` would outlast any client's tool timeout on a first rent (5 to 15 minutes). It starts setup as a background job, waits up to `wait_seconds` (45 by default), and reports `status: "running"` until the model answers. Calling it again re-attaches to the same job; it never starts a second machine. The job outlives the MCP server, so a client restarted mid-setup finds it again through `gpu_setup` or `gpu_status`, which answers during setup with `"busy": true` and the job's progress. `gpu_stop` and `gpu_destroy` cancel a setup in progress the way Ctrl-C does: the machine still booting is destroyed.

## What keeps the agent in line

- The skill tells the agent to show the price and ask before renting, wait out a slow setup instead of starting over, never touch the provider key, and offer stop or destroy (and check `gpu_ls`) when done.
- The server's tool annotations mark `gpu_destroy` destructive and the read-only tools read-only, so a client can let `gpu_status` run freely and ask for every `gpu_setup` and `gpu_destroy`. In Claude Code that is the `permissions` list in `.claude/settings.json`, e.g. `"allow": ["mcp__gpu__gpu_status", "mcp__gpu__gpu_offers", "mcp__gpu__gpu_ls"]`.
- `gpu.yaml` holds the hard limits the agent cannot exceed without editing the file: `budget.max_hourly_cost_usd`, `budget.max_session_cost_usd`, `lifecycle.max_lifetime`.
- The watchdog stops the machine after `idle_timeout` whatever the agent does, and `gpu sweep --install-timer` catches anything left behind.
