<p align="center">
  <a href="https://gpu4agents.jambu.ai">
    <img src="docs/assets/banner.png" alt="GPU for Agents: your agent rents the GPU it needs, on its own. You pay only while it works." width="100%">
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/gpu4agents/"><img src="https://img.shields.io/pypi/v/gpu4agents?color=5CBF3A&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/gpu4agents/"><img src="https://img.shields.io/pypi/pyversions/gpu4agents?color=5CBF3A" alt="Python versions"></a>
  <a href="https://github.com/jambuai/gpu4agents/actions/workflows/ci.yml"><img src="https://github.com/jambuai/gpu4agents/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-5CBF3A" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://gpu4agents.jambu.ai">Website</a> ·
  <a href="https://gpu4agents.jambu.ai/#video">1-minute video</a> ·
  <a href="docs/AGENTS.md">Agents guide</a> ·
  <a href="https://github.com/jambuai/gpu4agents/wiki">Wiki</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

# GPU for Agents

**Your agent rents the GPU it needs, on its own.**
**You pay only while it works: we watch every machine and shut down the idle ones.**

Ask Claude Code, Cursor or any MCP client to "compare Qwen 7B and Llama 8B on these 50 prompts". The agent finds a GPU under your budget, serves the model behind an OpenAI-compatible API with vLLM, and runs the experiment. A watchdog runs on the machine itself, so when nothing uses it for `idle_timeout` (15 minutes by default) it shuts down, even if the agent forgot, the chat closed or your laptop is asleep.

- **Self-service GPUs for agents.** Claude Code, Codex, Gemini CLI and Cursor install it themselves from one pasted line. A skill and local MCP tools (`gpu_offers`, `gpu_setup`, `gpu_run`, `gpu_status`, `gpu_destroy`, ...) give the agent the whole lifecycle. The skill tells it to show you the price and ask before renting.
- **Billing that ends when the work does.** Idle timeout, maximum lifetime, hourly and session budgets, and a sweep that cleans up anything left behind.
- **Any app, any language.** `gpu --json setup` returns `base_url`, `api_key` and `model` for any OpenAI SDK.

Vast.ai is the first provider; another provider is a new adapter, and the lifecycle rule stays the same.

## Quickstart: let your agent install it

Paste this into Claude Code, Codex, Gemini CLI or Cursor (say which one you use):

```text
Set up GPU for Agents for me. I'm using Claude Code. Fetch https://gpu4agents.jambu.ai/install.md with curl and follow its steps.
```

The agent installs the `gpu` command (with [uv](https://docs.astral.sh/uv/), which brings its own Python), the `gpu4agents` skill and the `gpu` MCP tools. Then you connect your GPU account in your own terminal, so the agent never sees the key:

```bash
gpu login        # paste your Vast.ai key (https://cloud.vast.ai/manage-keys/) into the hidden prompt
```

Restart the agent and ask for an experiment. It shows the price and asks before renting anything. See [Letting a coding agent rent GPUs](docs/AGENTS.md).

Prefer a terminal? These do the same:

```bash
curl -fsSL https://gpu4agents.jambu.ai/install.sh | sh          # uv + Python + gpu, then every agent it finds
npx skills add jambuai/gpu4agents                              # just the skill; it installs gpu on first use
uv tool install gpu4agents && gpu agent install -t codex --global   # by hand: claude | codex | gemini | cursor | auto
```

## Quickstart: from a shell or an app

```bash
gpu init --model Qwen/Qwen2.5-7B-Instruct
gpu validate
gpu offers                         # see prices, rent nothing
gpu setup                          # rent, boot vLLM, wait until the model answers
gpu run python experiment.py       # OPENAI_BASE_URL / OPENAI_API_KEY are set for the job
gpu destroy                        # or walk away: the idle timeout does it
```

From an application in any language, `gpu --json setup` prints the endpoint as JSON. See [Integrating into an application](docs/INTEGRATION.md), with a ready TypeScript client.

## Declare the limits

```yaml
# jambu.yaml (only the gpu_runtime key is read, so other Jambu tools can share the file)
gpu_runtime:
  provider: {name: vast}
  model_profile: qwythos-9b
  lifecycle:
    idle_timeout: 15m          # the watchdog on the machine stops it
    max_lifetime: 6h           # a hard ceiling, busy or not
  budget:
    max_hourly_cost_usd: 0.60  # offers above this are never rented
    max_session_cost_usd: 5.00
  health:
    pull_stall_timeout: 10m    # a host whose image pull stalls is replaced
```

A full example is [`jambu.example.yaml`](jambu.example.yaml). As a last line of defense, `gpu ls` shows every machine billing in the account and `gpu sweep --install-timer` guards every project on the machine and cleans up orphans. See [Keeping the account clean](docs/USAGE.md#keeping-the-account-clean).

`gpu login` saves the API key in `~/.gpu/credentials` (mode 600) after checking it; the environment and a project `.env` work too. It never goes in the manifest, and the agent never sees it.

## Docs

- [Letting a coding agent rent GPUs (Claude Code, Codex, Gemini CLI, Cursor, Claude Desktop)](docs/AGENTS.md)
- [Integrating into an application (any language)](docs/INTEGRATION.md)
- [Commands, workloads, and agents](docs/USAGE.md)
- [Every config field](docs/CONFIG.md)
- [Choosing a model](docs/CHOOSING_A_MODEL.md)
- [How shutdown works](docs/LIFECYCLE.md)
- [Adding a provider](docs/ADDING_A_PROVIDER.md)

## Coming from `jambu-gpu`

The package was called `jambu-gpu` until 0.4.1. Same code, same `gpu` command, same `jambu.yaml`, same machines:

```bash
pipx uninstall jambu-gpu && pipx install gpu4agents
```

Sweep timers installed by `jambu-gpu` keep working; `gpu sweep --install-timer` rewrites them for the new module.

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately, as described in [SECURITY.md](SECURITY.md).

## Author

Created by [João Gabriel Lima](https://www.linkedin.com/in/joaogabriellima) at [Jambu.ai](https://jambu.ai).

## License

[MIT](LICENSE).

<p align="center">
  <a href="https://jambu.ai">
    <img src="docs/assets/jambu-logo.jpg" alt="Jambu.ai — Applied Intelligence Lab" width="240">
  </a>
</p>
