# Changelog

All notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

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

[0.5.0]: https://github.com/jambuai/gpu4agents/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/jambuai/gpu4agents/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/jambuai/gpu4agents/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/jambuai/gpu4agents/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jambuai/gpu4agents/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jambuai/gpu4agents/releases/tag/v0.2.0
