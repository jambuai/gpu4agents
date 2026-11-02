# Changelog

All notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.7.0] - 2026-11-02

### Added
- Any GPU provider. An adapter can live in the project (`provider.plugin: ./gpu_providers/acme.py:AcmeProvider` in `jambu.yaml`) or in an installed package that declares a `gpu4agents.providers` entry point. Neither needs a change to gpu4agents. Vast.ai stays built in.
- `gpu providers new NAME` writes an adapter skeleton with every method that needs the provider's API marked. `gpu init --provider NAME` writes a `jambu.yaml` that loads it.
- [`PROVIDERS.md`](PROVIDERS.md), served at `gpu4agents.jambu.ai/providers.md`: the steps an agent follows to add a provider, from the skeleton to one test rental with the user's approval.
- The provider contract has `credentials_url` and `verify_credentials()`, which `gpu login` uses for every provider.
- Plugin manifests in the style of [Humanizer](https://github.com/blader/humanizer): a Claude Code marketplace (`/plugin marketplace add jambuai/gpu4agents`), a Cursor plugin, and Codex metadata in `skills/gpu4agents/agents/openai.yaml`. Each release attaches `gpu4agents-skill.zip` for Claude.ai and Claude Desktop.

### Changed
- `gpu login` works with any provider: `--provider NAME` picks it (default: the project's, else `GPU_PROVIDER`, else Vast.ai), it prompts for each credential the adapter declares, and it saves `GPU_PROVIDER` so account-wide commands outside a project use that provider. `--json` output now includes `provider`.
- The skill, `INSTALL.md`, the MCP server's instructions and the installer no longer assume Vast.ai. When no account is connected, the agent asks which provider the user wants.
- Docs and the website were edited for plain language.

## [0.6.0] - 2026-11-02

### Added
- **Agents install it themselves.** Paste one line into Claude Code, Codex, Gemini CLI or Cursor; the agent follows [`INSTALL.md`](INSTALL.md) (served at `gpu4agents.jambu.ai/install.md`) and sets everything up, without renting anything.
- `install.sh`: needs only `curl`. Installs uv if missing and `gpu4agents` with a Python uv downloads when needed, removes an old `jambu-gpu` tool, and runs `gpu agent install` for the runtimes it finds (`--agent` to pick one).
- `gpu login`: the user types the Vast.ai key into a hidden prompt; it is checked against the API, then saved in `~/.gpu/credentials` (mode 600). `--check` reports whether a working key is found and where; `--stdin` reads it from a pipe.
- `gpu agent install -t codex` (skill in `.agents/skills/`, MCP server in `~/.codex/config.toml`), `-t gemini` (skill in `.agents/skills/`, MCP server in `.gemini/settings.json`) and `-t auto` (every runtime found).
- The skill is published at `skills/gpu4agents/SKILL.md`, so `npx skills add jambuai/gpu4agents` installs it, and it now covers installation, connecting the account, monitoring (`status`, `logs`, `events`, `ls`, `sweep`) and the exit codes.

### Changed
- The skill is named `gpu4agents` (it was `gpu`). `gpu agent install` replaces the old `gpu` skill folder it wrote before; a folder of the same name from someone else is left alone.
- Cursor now gets the skill in `.agents/skills/` instead of `.claude/skills/`.

## [0.5.0] - 2026-11-02

### Changed
- **Renamed to GPU for Agents.** The package is now `gpu4agents` on PyPI (it was `jambu-gpu`) and the module is `gpu4agents`. The `gpu` command, the MCP server and tool names, `jambu.yaml`, `.gpu/`, `~/.gpu` and the labels on running instances are unchanged, so existing projects and machines keep working.
- The repository moved to [jambuai/gpu4agents](https://github.com/jambuai/gpu4agents); the old URLs redirect.
- New website: [gpu4agents.jambu.ai](https://gpu4agents.jambu.ai).

### Added
- `gpu --version` (same as `gpu version`).

### Migrating
- `pipx uninstall jambu-gpu && pipx install gpu4agents` (or `uv tool install gpu4agents`).
- A `jambu_gpu` alias module ships with the package, so sweep timers that run `python -m jambu_gpu sweep --apply` keep working; `gpu sweep --install-timer` rewrites them for `gpu4agents`.
- `jambu-gpu` 0.5.0 on PyPI is a transition release that installs `gpu4agents`.

## [0.4.1] - 2026-10-25

### Fixed
- A stalled image pull no longer costs 30 minutes per host. `gpu setup` replaces a host whose pull shows no progress for `health.pull_stall_timeout` (default `10m`; `off` disables it). The host is destroyed, remembered as bad for 24h, and another is rented. A host that pulls slowly but steadily still waits for `boot_timeout`.

## [0.4.0] - 2026-10-25

### Added
- `gpu mcp`: a local MCP server (stdio) with 13 tools, from `gpu_init` to `gpu_destroy`. `gpu_setup` runs in the background and re-attaches on each call, so a long first rent never hits a tool timeout.
- A bundled skill that tells the agent to show the price and ask before renting, wait out a slow setup, never touch `VAST_API_KEY`, and clean up.
- `gpu agent install` (`-t claude | cursor | desktop`, `--global`) writes the skill and registers the server.

### Fixed
- `gpu status` and `gpu logs` work during a setup (`"busy": true`); a held project lock is exit code 12.
- The last host that failed when setup gave up is remembered as bad.
- On Vast the vLLM image defaults to `vastai/vllm:v0.32.0-cuda-13.0`, cached on Vast hosts.

## [0.3.1] - 2026-10-25

### Fixed
- Vast: floor the GPU compute capability (vLLM: 8.0 on CUDA-13 images, 7.0 on CUDA-12), overridable with `compute.gpu.compute_cap_min`, so old cards behind a new driver are no longer rented.

## [0.3.0] - 2026-10-24

### Changed
- Files live in `.gpu/` (per project) and `~/.gpu` (per user) instead of `.jambu`. The first command renames the old directories. `GPU_HOME` overrides `~/.gpu`.

## [0.2.0] - 2026-10-24

### Added
- The watchdog retries a failed stop, re-arms on every restart and runs under a supervisor; inference traffic counts as activity; stopped instances are destroyed after `lifecycle.stopped_ttl`.
- `gpu ls` and `gpu sweep` (with a systemd timer) guard every project and clean up orphans.
- A per-instance endpoint key; `gpu --json setup` / `gpu env` return `base_url`, `api_key` and `model`.
- [docs/INTEGRATION.md](docs/INTEGRATION.md) and a one-file TypeScript client.

[0.6.0]: https://github.com/jambuai/gpu4agents/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jambuai/gpu4agents/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/jambuai/gpu4agents/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/jambuai/gpu4agents/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/jambuai/gpu4agents/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jambuai/gpu4agents/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jambuai/gpu4agents/releases/tag/v0.2.0
