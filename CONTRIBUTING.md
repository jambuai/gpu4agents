# Contributing

`gpu4agents` provisions a temporary GPU runtime and guarantees it stops billing
when nothing is using it. Contributions are welcome. By submitting a change you
agree it is licensed under the [MIT License](LICENSE).

## Command name

The published command is `gpu`. The package is `gpu4agents` (published as `jambu-gpu` before 0.5.0).

Do not add a second console script. One command, in `pyproject.toml`:

```toml
[project.scripts]
gpu = "gpu4agents.cli.main:main"
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Python 3.10 or newer. Tests do not need `VAST_API_KEY`, a GPU, or network access.

## Tests

```bash
.venv/bin/pytest
.venv/bin/pytest tests/test_policy.py       # shutdown rule
.venv/bin/pytest tests/test_watchdog.py     # generated guard, as a real process
.venv/bin/pytest tests/test_integration.py  # setup -> boot -> run -> auto-stop
```

A change to lifecycle behaviour needs a test that fails before the change and
passes after it. The policy in `gpu4agents/core/policy.py` and the watchdog that
runs on the instance must keep agreeing: the watchdog inlines that module at
provision time.

`tests/test_integration.py` does not mock the boot. It runs the onstart script
the engine generated, starts the real watchdog, waits on a real health endpoint,
runs a workload with real heartbeats, and asserts the guard stops the instance
with the right reason.

## What you may change

| Change | Where it goes |
| --- | --- |
| New provider | `gpu4agents/providers/<name>/` plus one registry line. See [docs/ADDING_A_PROVIDER.md](docs/ADDING_A_PROVIDER.md). |
| Shutdown rule | `gpu4agents/core/policy.py` and `tests/test_policy.py`. |
| How the model server starts | `gpu4agents/runtime/`. |
| CLI surface | `gpu4agents/cli/`. Keep `--json` and `--config` on every command. |

The core must not branch on a provider name. Branch on `provider.capabilities`.
If a provider feature requires an `if provider == "..."` in `core/`, the contract
is wrong. Extend the contract instead.

Out of scope: multi-instance scheduling, multi-GPU
orchestration, Kubernetes, distributed inference, automatic provider selection,
job queues, a web dashboard. See [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md).

## Secrets

Never commit credentials, `.env`, `.gpu/state.json`, or provider instance
ids from a real run. Keys are read from the environment, a gitignored `.env`,
or `~/.gpu/credentials`. `.env.example` ships with empty values only.

## Pull requests

- Explain why the behaviour should change, not only what moved.
- Run `.venv/bin/pytest` and report the result.
- Keep commits focused. A provider adapter and a policy change are two PRs.
- Match the surrounding style: English identifiers and docs, type hints on new
  public functions, no new dependency unless the standard library cannot do the job.

## Releasing

1. Bump `version` in `pyproject.toml` and `gpu4agents/__init__.py`, and add the entry to `CHANGELOG.md`.
2. Merge to `main`, tag `vX.Y.Z` and publish a GitHub release with the changelog entry as notes.
3. The `Release` workflow builds the wheel and sdist, checks the tag matches the version, and attaches them to the release. With the repository variable `PYPI_PUBLISH=true` and a PyPI trusted publisher for this repository (workflow `release.yml`, environment `pypi`), it also publishes to PyPI.
