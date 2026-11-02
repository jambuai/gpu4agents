# Letting a coding agent rent GPUs

`gpu` ships a skill and a local MCP server, so Claude Code, Codex, Gemini CLI, Cursor or Claude Desktop can rent a GPU, serve a model, run an experiment against it and shut it down, from a request like "compare Qwen 7B and Llama 8B on these 50 prompts".

The watchdog on the machine is what makes this safe to hand to an agent: a machine the agent forgets, or a conversation that ends mid-experiment, stops by itself after `lifecycle.idle_timeout`.

## Install: let the agent do it

Paste into the agent, naming it:

```text
Set up GPU for Agents for me. I'm using Codex. Fetch https://gpu4agents.jambu.ai/install.md with curl and follow its steps.
```

[`INSTALL.md`](../INSTALL.md) (served at `/install.md`) tells the agent to run [`install.sh`](../install.sh) with `--agent <runtime>`, check the account with `gpu login --check`, send the user to `gpu login` if there is no key, read the skill with `gpu agent skill`, and stop there: nothing is rented during installation.

`install.sh` needs only `curl` on Linux or macOS (Windows: WSL). It installs [uv](https://docs.astral.sh/uv/) if missing and runs `uv tool install --upgrade --python 3.12 gpu4agents`, so uv downloads a Python when the machine has no suitable one. It removes an old `jambu-gpu` tool, then runs `gpu agent install -t <runtime> --global` (`auto` by default: every runtime it finds). Options: `--agent claude|codex|gemini|cursor|auto|none`, `--no-modify-path`; `GPU4AGENTS_SPEC` installs a pinned version or a wheel.

With the [skills](https://skills.sh) CLI, `npx skills add jambuai/gpu4agents` installs only the skill (from [`skills/gpu4agents/`](../skills/gpu4agents/SKILL.md)); its first section installs the `gpu` command when it is missing.

## Connect the account: `gpu login`

The user, in their own terminal:

```bash
gpu login            # hidden prompt; checks the key against Vast.ai, then saves it
gpu login --check    # exit 0 and where the key was found, or exit 5
echo "$KEY" | gpu login --stdin
```

The key goes to `~/.gpu/credentials` (mode 600, directory 700); other lines in the file are kept. A key in the environment or a project `.env` still works and wins over the file. The skill tells the agent never to ask for the key in chat, read those files, or print it.

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

Claude Desktop and Cursor do not get your shell's environment. Keep `VAST_API_KEY` in `~/.gpu/credentials` (`gpu login`) or the project `.env`, not only in an `export`.

Another agent that reads `SKILL.md` files: `gpu agent skill > path/to/skills/gpu4agents/SKILL.md`. Any MCP client: run `gpu mcp --project /path/to/project` over stdio.

## The MCP tools

| Tool | Does | Costs money |
| --- | --- | --- |
| `gpu_init` | write `jambu.yaml` for a model, with an optional price cap and VRAM | no |
| `gpu_validate` | check the config and credentials | no |
| `gpu_offers` | cheapest matching machines and their price | no |
| `gpu_setup` | rent / restart and wait for the model; returns `base_url`, `api_key`, `model` | yes |
| `gpu_status` | state, cost so far, idle time, the watchdog's next move, setup progress | no |
| `gpu_connection` | `base_url`, `api_key`, `model` without starting anything | no |
| `gpu_run` | run a local command with `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `JAMBU_MODEL_ID` set | yes, if it has to start the machine |
| `gpu_logs` | runtime, onstart, watchdog or provider log | no |
| `gpu_lock` / `gpu_unlock` | keep the machine up through a quiet period | while locked |
| `gpu_stop` | stop now; disk kept, restart takes about a minute | ends billing for compute |
| `gpu_destroy` | delete; needs `confirm=true` | ends all billing |
| `gpu_ls` | every machine in the account, from any project | no |

Each tool runs the `gpu` CLI with `--json` in the project directory, so errors carry the same `exit_code` as the [application contract](INTEGRATION.md#exit-codes), plus `meaning` and `next` for the agent. A tool takes an optional `project_dir`; by default it is the server's `--project`, `$GPU_PROJECT_DIR`, or the directory the client started it in.

`gpu_setup` would outlast any client's tool timeout on a first rent (5–15 minutes). It starts setup as a background job, waits up to `wait_seconds` (45 by default), and reports `status: "running"` until the model answers. Calling it again re-attaches to the same job; it never starts a second machine. The job outlives the MCP server, so a client restarted mid-setup finds it again through `gpu_setup` or `gpu_status`, which answers during setup with `"busy": true` and the job's progress. `gpu_stop` and `gpu_destroy` cancel a setup in progress the way Ctrl-C does: the machine still booting is destroyed.

## What keeps the agent in line

- **The skill** tells the agent to show the price and ask before renting, wait out a slow setup instead of starting over, never touch `VAST_API_KEY`, and offer stop or destroy (and check `gpu_ls`) when done.
- **The server's tool annotations** mark `gpu_destroy` destructive and the read-only tools read-only, so a client can let `gpu_status` run freely and ask for every `gpu_setup` and `gpu_destroy`. In Claude Code that is the `permissions` list in `.claude/settings.json`, e.g. `"allow": ["mcp__gpu__gpu_status", "mcp__gpu__gpu_offers", "mcp__gpu__gpu_ls"]`.
- **`jambu.yaml`** holds the hard limits the agent cannot exceed without editing the file: `budget.max_hourly_cost_usd`, `budget.max_session_cost_usd`, `lifecycle.max_lifetime`.
- **The watchdog** stops the machine after `idle_timeout` whatever the agent does, and `gpu sweep --install-timer` catches anything left behind.
