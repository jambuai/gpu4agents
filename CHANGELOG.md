# Changelog

All notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`gpu gateway`: a managed local endpoint that follows the machine.** Provisioning a model is only half the job - you still have to reach it, and the rented machine's `base_url` changes every time it is re-rented. `gpu gateway start` runs a loopback endpoint (127.0.0.1) that forwards to the current instance, re-reading the endpoint from state so it keeps working across stop/restart. `gpu gateway install` writes a systemd user service (`Type=simple`, `Restart=always`, boot-safe via linger) so the gateway owns its own lifecycle instead of being a hand-run script. Pluggable adapters shape the request per client: `openai`, `anthropic` (drops the client's own credential so a claude.ai OAuth token never reaches the rented machine, injects the instance key, renames the model to the served one) and `none` (transparent), plus a `gateway.plugin` escape hatch mirroring provider plugins. The `anthropic` adapter also normalizes reasoning effort across `reasoning_effort`, `reasoning.effort` and `output_config.effort`, mapping to what the served model accepts, so a client that spells an effort level differently no longer 400s. New `gateway:` section in gpu.yaml; off by default.

## [0.14.0] - 2026-11-12

### Added
- **Queue mode** for applications that send jobs over the day. A `queue:` section in `gpu.yaml` turns it on; a project without one behaves exactly as 0.13.0. Full guide: [QUEUE.md](QUEUE.md).
  - `gpu queue submit | status | ls | cancel | retry | pull | pause | resume | drain | audit | stats | run | install-service`, a Python API (`gpu4agents.queue.api`) and the MCP tools `gpu_queue_submit`, `gpu_queue_status`, `gpu_queue_cancel`, `gpu_queue_pull` and `gpu_queue_control`. Every control action is recorded with who did it.
  - A **coordinator** (`gpu queue run`) opens machines when the oldest queued job has waited `scale_up_after`, within `max_machines` and `max_usd_per_hour`, checks each machine's health, runs jobs back to back on a warm machine, and retires machines that have nothing to do. A second coordinator waits as a standby.
  - **Retries**: a job whose machine vanished, went silent or failed its health check runs again on another machine, up to `max_attempts`, then fails with every attempt's reason. A retried attempt finds the earlier attempt's outputs in `$JAMBU_PREVIOUS_OUTPUT_DIR`.
  - **Deadlines instead of lifetimes**: the coordinator moves each machine's deadline in its own watchdog (`POST /deadline`) to cover the job it took. With the coordinator dead, the pool drains to zero by itself.
  - **Preflight** per job (`timeout`, and the new optional `max_cost_usd` and `min_vram_gb` in `gpujob.yaml`), or your own `module:function`.
  - **Backends**: `local` (SQLite) and `redis` (Redis 5 or newer, one node), and outputs to a local directory or S3, each checked against a sha256 manifest. Others plug in through [the contract](docs/QUEUE_BACKENDS.md); `QueueBackendContract` is the suite a backend runs. Addresses and keys come from `GPU_QUEUE_URL`, `GPU_OUTPUTS_URL` and `AWS_*` in the environment or `~/.gpu/credentials`, never `gpu.yaml`. `redis` and `boto3` are optional extras (`queue-redis`, `queue-s3`).
  - **An auditor** (`gpu queue audit`) checks eight invariants (I1 to I8) and repairs what is safe; events, metrics and alert thresholds are in QUEUE.md.
- `gpu sweep` and the sweep timer cover pool machines, so a dead coordinator still ends in a destroy with finished jobs' outputs saved, and a pool machine is never taken for an orphan.
- Acceptance guarantees 24 to 31, one for each thing queue mode must never do (docs/ACCEPTANCE.md).

### Notes
- Set `queue.namespace` when you use Redis: without it the keyspace follows the config fingerprint and changes when the runtime part of `gpu.yaml` changes.
- After a takeover by a coordinator whose pool is different, a job on a machine it cannot see waits until that attempt's deadline plus a grace before it runs again.
- Redis Cluster is not supported.

## [0.13.0] - 2026-11-11

The first real, filmed use of 0.12.0 (an agent making images with SDXL and running a CUDA benchmark) found these. Each is fixed and tested.

### Fixed
- **A machine stopped by hand no longer bills forever without a word.** The agent ended its session with `gpu stop`; on Vast.ai that keeps billing the disk, and with no sweep timer installed nothing would ever destroy it.
  - `gpu stop` and the MCP `gpu_stop` say so: `still_billing`, the disk's cost per day, and whether the sweep timer will remove it after `stopped_ttl`.
  - Every `gpu` command, in any directory, warns about a registered machine left stopped and billing. It reads state files only, so it costs no API call.
  - `gpu agent install` asks once to install the sweep timer (`--sweep-timer` / `--no-sweep-timer`). The timer is the only thing that enforces `stopped_ttl` on a stopped machine.
  - The skill now says to destroy once the work is done and the results are home, the same choice the watchdog makes. Stop only when the user will be back soon or keeps the disk on purpose.
- **The card the user approved is the one rented.** The agent quoted an RTX 3060 at about $0.08/h and got an RTX 2060 at $0.094/h when the quoted offer was taken. `gpu offers` remembers what it showed, and `gpu setup --offer ID` (MCP `gpu_setup(offer_id)`) rents only that card at no more than that price, host retries included, or nothing. Every setup reports `rented`, `quoted` and `matches_quote`, and warns when they differ.
- **A project is never kept where it will be lost.** The agent kept `gpu.yaml` in its session scratchpad; the next session saw the machine as an orphan it could not use. `gpu init` refuses a temporary or scratchpad folder (`--allow-temp` overrides), and the skill puts the project at the root of the user's repository.
- **A machine that vanished leaves a reason.** The log said only "instance missing at provider". Now the reconcile replays the project's rules on what it last knew and records `instance_destroyed reason=idle_timeout by=watchdog inferred=true` when they called for it, or `likely_reason: unknown` when they did not. `gpu setup` also fetches the watchdog's own events as soon as the machine is ready.

### Added
- `gpu adopt <instance-id>` (MCP `gpu_adopt`) takes over an orphan this account owns. On Vast.ai the boot script the provider kept gives back its watchdog token and model key, so `gpu setup` reuses it when gpu.yaml matches; otherwise it can still be destroyed from there. The provider key in that script is never read back.
- `gpu offers --for <minutes on an RTX 4090>` (MCP `for_minutes_on_4090`) ranks offers by the estimated cost of the whole job, not the price per hour.
- Image models are sized for their peak: `gpu init` gives SDXL 16 GB and FLUX 24 GB, and `gpu validate` warns below that (a 12 GB card ran out of memory with SDXL at 1344x768). `gpu.models.yaml` has `sdxl-base`, `flux-dev` and `flux-schnell` profiles.

### Changed
- The skill and the MCP instructions: size for the peak with 1.5 to 2 times headroom; stop a server before a job that shares its card; compare offers per job; say that `idle_timeout` is 15 minutes by default and the user's to change; the only credentials check is `gpu login --check`, never listing environment variables or a secret store, never another provider's key.

## [0.12.0] - 2026-11-07

### Fixed
- **A job's results are no longer deleted before anyone downloaded them.** 0.11.0 made an idle Vast.ai machine be destroyed, and with it the outputs of a job run with `--detach` that nobody had pulled yet, or the checkpoints of a job cut off by `max_lifetime`. Now:
  - On the machine, the job runner knows which outputs were downloaded after their job ended. While any are not, a shutdown that would delete them becomes a stop that keeps the disk. Where no stop keeps the disk (Lambda, a RunPod pod without a volume), an idle machine waits up to 2 hours for the download.
  - `gpu sweep` and `gpu guard` download finished jobs' results into `gpu-outputs/<job id>/` on their own, and before any automatic destroy. A stopped machine that still holds results (and bills its disk) is started, emptied and destroyed. With `gpu sweep --install-timer`, results reach your computer within minutes, with nobody at the keyboard.
  - `gpu job pull` works on a stopped machine: it starts it, downloads, and puts it back.
  - `gpu destroy` (and the MCP `gpu_destroy`) downloads the results first, and refuses when it cannot; `--discard-outputs` (`discard_outputs=true`) deletes them.
  - `gpu status` lists the jobs whose results are only on the machine (`job_outputs_unsaved`).
- `gpu job run` refuses a job whose `timeout` is longer than `lifecycle.max_lifetime` before anything runs, instead of letting the machine shut down mid-job, and warns when the machine will shut down before a job without a timeout could finish.

## [0.11.0] - 2026-11-06

### Changed
- **Where a stopped machine still bills storage, the guard destroys it instead of stopping it.** That is Vast.ai, and a RunPod pod with a volume. A stopped Vast.ai machine kept billing its disk (an 80 GB disk is about $2.30 a day), and nothing on a stopped machine can remove it: `lifecycle.stopped_ttl` was enforced only by `gpu guard` or `gpu sweep` on your computer, which on a Mac never ran on its own. This applies to idle, `max_lifetime` and the budget. The next `gpu setup` rents a new machine and downloads the model again (a few minutes instead of about one). `lifecycle.keep_disk: true` restores the old behavior for those who want the fast restart and accept the storage bill.

### Added
- `gpu sweep --install-timer` on macOS writes a LaunchAgent (`~/Library/LaunchAgents/ai.jambu.gpu-sweep.plist`) that runs the sweep every 5 minutes and at login, so `stopped_ttl` and orphan cleanup work on a Mac too. Linux keeps the systemd user timer.

### Fixed
- When the CLI's "model is up" signal did not reach the watchdog (after a restart, for instance), the idle clock did not start until the startup grace (about 45 minutes) ran out. A machine left idle could bill for over an hour instead of `idle_timeout`. The watchdog now ends the startup by itself as soon as the model server answers on the machine.

## [0.10.1] - 2026-11-06

### Fixed
- A dropped connection while vLLM downloads the model no longer costs the machine. vLLM dies on such an error, and gpu used to destroy the host, rent another and pull the image again; three in a row ended the setup. Now the model server is restarted on the same machine, up to 4 times with a growing pause, and the download resumes from the files already on disk. Only a network that keeps failing counts as a host failure.
- On hosts where the container sees many CPUs but may start few processes (a low `pids.max`), thread pools sized to the visible cores (the Hugging Face downloader, OpenMP, torch) failed to start and crashed the download or the server. The model server now runs with those pools capped to the CPU quota on such hosts, as Vast.ai's own images do. Values you set yourself are kept.

### Changed
- The vLLM launcher travels compressed in the boot script, which is close to Vast.ai's 16 KB limit.

## [0.10.0] - 2026-11-06

### Added
- **RunPod and Lambda are built in**, next to Vast.ai. `gpu login` lists all three, with the steps for each in plain words, and saves the one you pick; `gpu init` then writes it into `gpu.yaml` (`--provider` still overrides). The watchdog, budgets, `gpu ls`, `gpu sweep` and the MCP tools work the same on each.
  - RunPod (`RUNPOD_API_KEY`) uses the REST API v2. A pod runs the model image with gpu's boot script; the model and watchdog are reached through RunPod's HTTPS proxy. Without a volume (`provider.options.volume_gb`, default 0) a stopped pod costs nothing. RunPod's API has no spot pods, and gpu does not SSH into pods: run code on the GPU with `gpu job`.
  - Lambda (`LAMBDA_API_KEY`) rents whole VMs. gpu runs the model image on the VM with Docker, registers its own SSH key (kept in `$GPU_HOME/ssh/`) and creates a firewall ruleset per region for the model and watchdog ports. Lambda cannot stop a machine, so where another provider would stop, gpu and the watchdog terminate it. Instance types have a fixed GPU count (1, 2, 4 or 8).
- RunPod and Lambda report a GPU's model but not its architecture, so gpu looks it up (`providers/gpus.py`) and skips cards too old for the model image, as it already did on Vast.ai.

### Changed
- `gpu init` without `--provider` uses the provider `gpu login` connected, instead of always Vast.ai.
- `health.pull_stall_timeout` applies only to providers that report the image pull moving while a machine boots (the new `boot_progress` capability; Vast.ai does). On the others, a long pull is waited out until `health.boot_timeout`, instead of being taken for a stall after ten quiet minutes.
- The website and docs: a table of how the three providers differ, at `/docs#providers`, and the adapter examples now use a provider that is not built in.

## [0.9.2] - 2026-11-06

### Changed
- `gpu login` starts by showing the providers you can use: Vast.ai, ready out of the box, or "another provider I already use", which hands the job to your agent (it writes the adapter). For Vast.ai it then lists the steps in plain words, from creating the account and adding credit to copying the key, before the hidden prompt. `--provider`, `--stdin` and a project's `gpu.yaml` skip the menu, as before.
- The messages a user sees when no account is connected, from `gpu login --check`, the installer, the skill and `install.md`, no longer assume a technical reader.

### Fixed
- Oh My Zsh's git plugin, on by default, defines `gpu` as an alias for `git push upstream`, and an alias wins over the command, in a terminal and in an agent's shell. The installer now detects a `gpu` alias in zsh and bash and removes it for new shells, in a marked block at the end of `~/.zshrc` or `~/.bashrc` (`--keep-gpu-alias` leaves it). The command is also installed as `gpu4agents`, which no alias takes, and the skill tells the agent to check `type gpu` first and use `command gpu` when it is an alias.

### Added
- Provider adapters can describe their own first-time walk-through for `gpu login`: `tagline`, `signup_url` and `setup_steps`.

## [0.9.1] - 2026-11-04

### Changed
- The config file is `gpu.yaml`, and the model catalog is `gpu.models.yaml`. `gpu init` writes `gpu.yaml`, and the docs, the skill and the website use the new names.

### Upgrading
- A project with a `jambu.yaml` or `jambu.models.yaml` keeps working: both names are still read, after the new ones. Renaming the file is enough to move over. In a directory that has a `jambu.yaml`, `gpu init` stops and asks you to rename it, or writes a `gpu.yaml` beside it with `--force`.

## [0.9.0] - 2026-11-04

### Added
- **Serve any Hugging Face model.** `gpu init --model <repo>` reads the Hugging Face model card and picks the server: chat and vision-language models stay on vLLM; every other task runs on gpu's own OpenAI-compatible server (`runtime.engine: hf`): `/v1/embeddings`, `/v1/images/generations`, `/v1/audio/transcriptions` and `/translations`, `/v1/audio/speech`, and `/v1/pipeline` for any other transformers pipeline. `model.task` records which; `--engine` and `--task` override the card, and `runtime.packages` adds pip packages a model needs.
- `runtime.engine: custom`: a model server the agent writes. `gpu init --engine custom` scaffolds `server/server.py` and `server/gpujob.yaml`; gpu uploads and runs it like the built-in one. [`SERVING.md`](SERVING.md), served at `gpu4agents.jambu.ai/serving.md`, is the contract.
- Both run as *service jobs* on the job runner: deployed after boot, so changing the model or the server's code and running `gpu setup` swaps the server on the same machine instead of renting a new one. `gpu redeploy` restarts it.
- Requests to these servers count as activity through their `/metrics`, so a machine in use stays up and an unused one stops after `idle_timeout`. A server that dies while loading ends `gpu setup` with its log, and `gpu logs` shows it.
- `gpu env --format json`, `gpu setup --json` and the MCP connection now include `engine`, `task` and `routes`. `gpu_init` takes `engine` and `task` and reports what it chose.
- `GPU_NO_MODEL_CARD=1` makes `gpu init` skip reading the model card (offline).

## [0.8.0] - 2026-11-04

### Added
- **Jobs: your own code on the GPU.** `gpu job run` packs a directory, uploads it to the rented machine, runs a shell command there, streams the log and downloads whatever the job wrote to `$JAMBU_OUTPUT_DIR` into `gpu-outputs/<job id>/`. Use it for anything that needs CUDA and is not a request to a served model: training or fine-tuning, image, audio or video models, batches, benchmarks. `gpu job status | logs -f | pull | cancel | ls` follow it.
- A running job counts as work: the watchdog on the machine heartbeats it, so the machine stays up with no laptop connected and becomes idle when the job ends. `max_lifetime` and the session budget still apply. Jobs survive a watchdog respawn; their exit code is written by the job's own shell.
- `gpujob.yaml` next to the code declares `run`, `setup`, `env`, `timeout`, `exclude` and `label`. `.git`, virtualenvs, caches, `.env` files and earlier outputs never upload.
- [`JOBS.md`](JOBS.md), served at `gpu4agents.jambu.ai/jobs.md`: the contract an agent follows to write a job.
- `gpu init --jobs` writes a `jambu.yaml` for a machine that serves no model (`runtime.engine: none`, PyTorch 2.8 with CUDA 12.8). `model` is now optional when nothing is served. `--min-vram-gb` sets the GPU size.
- MCP tools `gpu_job_run`, `gpu_job_status`, `gpu_job_pull` and `gpu_job_cancel`.
- The watchdog loads extensions the CLI uploads once the machine is up (`POST /extend`), so the boot script stays under Vast.ai's 16 KB limit. The job runner is the first one.

### Changed
- The boot payload is base85 instead of base64, about 700 bytes smaller.
- The default image for `runtime.engine: none` is `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (was 2.4.0 with CUDA 12.1).

### Fixed
- A machine with `runtime.engine: none` stayed in "starting" until the boot grace ran out (about 45 minutes), so an unused one could not idle out sooner.

### Upgrading
- Machines rented by an earlier version have a watchdog that cannot load the job runner. `gpu job run` says so; `gpu setup --recreate` replaces the machine.

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
