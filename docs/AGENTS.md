# Letting a coding agent rent GPUs

`gpu` ships a skill and a local MCP server, so Claude Code, Cursor or Claude Desktop can rent a GPU, serve a model, run an experiment against it and shut it down, from a request like "compare Qwen 7B and Llama 8B on these 50 prompts".

The watchdog on the machine is what makes this safe to hand to an agent: a machine the agent forgets, or a conversation that ends mid-experiment, stops by itself after `lifecycle.idle_timeout`.

## Install

```bash
pipx install gpu4agents            # or: uv tool install gpu4agents
cd your-project
echo "VAST_API_KEY=..." >> .env   # or ~/.gpu/credentials; never in jambu.yaml
gpu agent install                 # Claude Code
gpu agent install -t cursor       # Cursor
gpu agent install -t desktop      # Claude Desktop
```

| Target | Skill | MCP server |
| --- | --- | --- |
| `claude` (default) | `.claude/skills/gpu/SKILL.md` | `.mcp.json` (Claude Code asks once to approve it) |
| `cursor` | the same `.claude/skills/gpu/` (Cursor reads Claude's skill directories) | `.cursor/mcp.json`, with `--project ${workspaceFolder}` |
| `desktop` | `~/.gpu/gpu-skill.zip`, to upload under Settings > Capabilities > Skills | `claude_desktop_config.json`, pinned to this project |

`--global` installs for every project of the user instead: the skill in `~/.claude/skills/gpu/`, the MCP server through `claude mcp add --scope user` and in `~/.cursor/mcp.json`. Existing config files are merged; only the `gpu` entry is written. Re-running is safe.

Claude Desktop and Cursor do not get your shell's environment. Keep `VAST_API_KEY` in the project `.env` or `~/.gpu/credentials`, not only in an `export`.

Another agent that reads `SKILL.md` files: `gpu agent skill > path/to/skills/gpu/SKILL.md`. Any MCP client: run `gpu mcp --project /path/to/project` over stdio.

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
