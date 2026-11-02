# jambu.yaml reference

`jambu.yaml` is the source of truth for the desired state. It describes what should
run and how long it may live. It must never contain credentials, and it never contains
provider API details.

`jambu.yaml` is a manifest shared across Jambu.ai Lab sub-projects. `gpu` reads only
its own `gpu_runtime:` top-level key and ignores every other key in the document, so
other Jambu tools can keep their own config in the same file:

```yaml
# jambu.yaml
gpu_runtime:
  version: 1
  model: {id: ...}
  ...

some_other_jambu_tool:
  ...      # not read by gpu, and gpu never errors on it
```

If the file does not need to be shared, `gpu init --flat` writes a `config.yml` where
the whole document is the gpu-runtime config, with no `gpu_runtime:` wrapper. This
format is still supported. A bare document with no `gpu_runtime:` key is also accepted
under the `jambu.yaml` filename, for a solo project that prefers that name.

Resolution order (first match wins, searched from the working directory upward):
`--config PATH`, else `jambu.yaml`, `jambu.yml`, `config.yml`, `config.yaml` in that
order. When a directory has both shapes, the shared, namespaced one is preferred over
the legacy flat one.

`${VAR}` references in string values are expanded from the environment.

Unknown keys are a hard error, so a typo in `idle_timeout` cannot silently disable
your cost protection. This applies within the `gpu_runtime:` section. `gpu` does not
check keys that other tools own elsewhere in the document.

### `model_profile`: share model config across files

Set `model_profile: <name>` instead of declaring `model:`/`compute:`/`runtime:`
directly, and gpu merges in that named entry from `jambu.models.yaml`. The catalog is
searched the same way as `jambu.yaml` (this directory, then parents), so one catalog
serves every sub-project in a repo. Fields you declare locally still win over the
profile's. See
[`docs/CHOOSING_A_MODEL.md`](CHOOSING_A_MODEL.md#dont-write-a-new-jambuyaml-per-model---use-the-catalog)
for why this exists and `gpu profiles` / `gpu inspect-model
--add-profile` to manage the catalog.

## `provider`

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | `vast` | Adapter to use. `gpu providers` lists what is available. |
| `profile` | `default` | Credential profile. Looks up `VAST_API_KEY__<profile>` before `VAST_API_KEY`. |
| `options` | `{}` | Provider-specific tuning, validated by the adapter (see below). |

### `provider.options` for vast

| Field | Default | Meaning |
| --- | --- | --- |
| `api_base` | `https://console.vast.ai/api/v0` | API endpoint. |
| `runtype` | `ssh_direc ssh_proxy` | Vast container run mode. |
| `verified_only` | `true` | Only rent verified machines. |
| `min_reliability` | `0.95` | Minimum machine reliability score. |
| `min_inet_down_mbps` | `500` | Minimum download bandwidth. The vLLM image alone is ~10 GB and a slow pull is the commonest boot failure; most offers clear 500 at the same price. |
| `min_cuda` | `12.1` | Floor for the host's `cuda_max_good` (the newest CUDA its driver runs). The search uses the higher of this and `compute.gpu.cuda_min`, and offers are re-checked client-side. |
| `search_limit` | `24` | How many offers to consider. |
| `order_by` / `order_dir` | `dph_total` / `asc` | Offer ranking, cheapest first. |
| `image_login` | – | `-u user -p pass docker.io` for a private image. |
| `search_filters` | – | Raw extra filters merged into the offer query. |

## `compute`

`gpu inspect-model <repo>` reads a model's HuggingFace metadata and suggests
values for this section. [`docs/CHOOSING_A_MODEL.md`](CHOOSING_A_MODEL.md) lists
which source each number comes from.

| Field | Default | Meaning |
| --- | --- | --- |
| `gpu.min_vram_gb` | `24` | Minimum VRAM per GPU. |
| `gpu.count` | `1` | Number of GPUs on one machine. |
| `gpu.name_filter` | – | Substring match, e.g. `A100`, `RTX 4090`. |
| `gpu.cuda_min` | – | Minimum host CUDA version. `gpu` raises it to what the runtime image needs, read from its tag (`-cu129` means 12.9, vLLM's bare `v0.32+` and `latest` mean 13.0), so an image never lands on a driver too old to run it. |
| `gpu.compute_cap_min` | – | Minimum GPU compute capability (architecture), e.g. `8.0` for Ampere. Distinct from `cuda_min`: a Pascal (GTX 10xx, 6.1) or Volta (V100, 7.0) card can sit behind a modern CUDA-13 driver, and so clear `cuda_min`, yet still be too old for a runtime image whose kernels are built for newer architectures. The image then crashes at engine startup. `gpu` raises this to what the runtime needs: a CUDA-13 vLLM image floors it at `8.0` (those wheels drop Volta), a CUDA-12 vLLM image at `7.0`. The cheapest Vast offers are often these old cards, so without the floor the search keeps picking a GPU the image cannot run. |
| `disk_gb` | `80` | Disk. Must fit the model weights plus the container. |
| `region` | – | Comma-separated geolocations. |
| `spot` | `false` | Use interruptible/bid capacity where supported. |
| `host_attempts` | `3` | How many machines `setup` tries. A host that fails to boot (the provider reports it failed, it vanishes, or it is not running within `health.boot_timeout`) is destroyed, replaced from fresh offers (each held to the budget), and remembered in `$GPU_HOME/bad_hosts.json` so no setup rents it again for 24h. A model server that dies once its container runs fails at once, since the same config would fail on any machine, unless its log shows the host's network failed it (a dropped connection or timeout while downloading weights): that counts as a host failure too. Needs `lifecycle.cleanup_on_setup_failure`; without it the first failed host is kept for inspection. |
| `image` | – | Override the runtime's container image. vLLM's default is `vllm/vllm-openai:v0.32.0`, except on Vast, where it is `vastai/vllm:v0.32.0-cuda-13.0`. Pin a tag: `validate` warns on `latest` or an untagged image, because the CUDA it needs can change under you. The Vast default (the same vLLM, packaged by Vast) is cached on many hosts, where the 12 GB Docker Hub image often stalled past `health.boot_timeout`. |

## `model`

| Field | Default | Meaning |
| --- | --- | --- |
| `id` | *required* | HuggingFace repo id. |
| `revision` | – | Pin a commit/branch/tag. |
| `hf_token_env` | `HF_TOKEN` | Env var holding the HF token for gated models. |
| `trust_remote_code` | `false` | Pass `--trust-remote-code` to the runtime. |
| `served_name` | – | Name the model is served under (`--served-model-name`). |

## `runtime`

How the model runs. Independent from where it runs.

| Field | Default | Meaning |
| --- | --- | --- |
| `engine` | `vllm` | `vllm` or `none` (bare compute, start nothing). |
| `port` | `8000` | Container port the server listens on; published publicly. |
| `context_length` | `16384` | `--max-model-len`. |
| `dtype` | `auto` | vLLM dtype. |
| `gpu_memory_utilization` | `0.90` | Fraction of VRAM vLLM may claim. |
| `tensor_parallel_size` | `compute.gpu.count` | Shards across GPUs. |
| `api_key` | generated | The key the model endpoint requires. Unset, `setup` generates a random one per instance; `gpu env` prints it. |
| `vllm_args` | `{}` | Structured escape hatch for any vLLM flag this schema doesn't model: `{flag: value}` → `--flag value` (`true`/`false` → a bare/negated flag, a dict or list → JSON-encoded). Applied before `extra_args`. See [`docs/CHOOSING_A_MODEL.md`](CHOOSING_A_MODEL.md#dynamic--un-modeled-vllm-flags). |
| `extra_args` | `[]` | Raw argv appended verbatim, last, for anything `vllm_args` can't express. |
| `tool_calling` | `auto` | `auto` guesses `--tool-call-parser` from `model.id` and turns tool calling on if recognized; `on` requires a resolvable parser (fails `validate` otherwise); `off` never passes tool-calling flags. |
| `tool_call_parser` | – | Explicit vLLM parser name, overrides the guess. See [vLLM's tool calling docs](https://docs.vllm.ai/en/latest/features/tool_calling/) for the current list (`hermes`, `llama3_json`, `llama4_pythonic`, `mistral`, `granite`, `qwen3_xml`, `deepseek_v3`, ...). |
| `tool_server` | – | MCP server URL (or `demo`). vLLM routes `/v1/responses` tool calls to it and runs them server-side. Leave unset to let the calling agent framework execute tools itself. |

`gpu inspect-model <repo>` also suggests `dtype`, `context_length` and
`tool_calling` for a given model. See
[`docs/CHOOSING_A_MODEL.md`](CHOOSING_A_MODEL.md).

### Agentic flows: nothing to build

vLLM's OpenAI-compatible server already implements both agent-facing APIs.
`gpu` turns on the right flags and gives you the same endpoint as always:

- `/v1/chat/completions` with `tools=[...]` works with `tool_calling: auto`
  or an explicit `tool_call_parser`, like any OpenAI-compatible tool-calling
  setup (LangChain, LangGraph, CrewAI, the raw `openai` SDK, ...).
- `/v1/responses` is the OpenAI Responses API, byte-compatible with the
  official `openai` Python/JS SDK's `client.responses.create(...)`. The route
  needs no config to be exposed. To use tools it needs the same tool-call parser
  as chat completions, and `tool_server` if you want vLLM to execute MCP tools
  itself instead of your agent framework.

`gpu run` exports `JAMBU_GPU_ENDPOINT`, `JAMBU_MODEL_ID` and
`OPENAI_API_KEY`, and that is the whole integration. Any OpenAI-SDK-compatible
agent framework works against the instance gpu provisioned.
[`examples/mastra-agent/`](../examples/mastra-agent/) is a full installable
[Mastra](https://mastra.ai) project built on those three env vars.
`gpu validate` and `gpu status` report which parser is active
(`vLLM :8000 (model, tools=hermes)`). If no parser is active for a model you
expect to call tools, set `runtime.tool_call_parser` explicitly.

## `lifecycle`

The cost protection. See [the safety invariant](#safety-invariant).

| Field | Default | Meaning |
| --- | --- | --- |
| `max_lifetime` | `6h` | Hard ceiling on instance age. `null` = unbounded. |
| `idle_timeout` | `15m` | Shut down after this long with no activity. `null` = unbounded. |
| `stopped_ttl` | `24h` | Destroy an instance that has been `STOPPED` this long. A stopped Vast instance no longer bills GPU time but still bills storage. `null` keeps stopped instances forever. |
| `auto_stop` | `true` | The guard may stop the instance. |
| `auto_destroy` | `false` | The guard destroys instead of stopping. |
| `cleanup_on_setup_failure` | `true` | Destroy partially created resources when setup fails. |
| `allow_indefinite_lock` | `false` | Permit `gpu lock` with no TTL. |
| `heartbeat_interval` | `30s` | How often a running workload reports in. |
| `heartbeat_grace` | `5m` | Missing heartbeats tolerated before a workload is stale. |
| `ssh_counts_as_activity` | `false` | An SSH session alone is not useful work. |
| `traffic_counts_as_activity` | `true` | Requests served by the model runtime count as activity. The watchdog reads vLLM's `/metrics` on the instance (requests running or waiting, completed requests, prompt tokens), so an endpoint in use by something other than `gpu run` is not idled out. `false` = only workloads, startup and locks count. |
| `lock.enabled` | `false` | Whether locks are offered in this project. |
| `lock.default_ttl` | `2h` | TTL used by `gpu lock` with no `--for`. |
| `watchdog.enabled` | `true` | Install the independent remote guard. |
| `watchdog.port` | `8777` | Control port published on the instance. |
| `watchdog.interval` | `60s` | How often the remote guard evaluates the policy. |

At least one of `idle_timeout` / `max_lifetime` must be set, or `validate` fails.

`idle_timeout` may not exceed `max_lifetime`.

`max_lifetime` and `budget.max_session_cost_usd` are measured per session: from
creation, and again from every restart of a stopped instance. A restart gets a fresh
bound instead of being killed for the hours it spent stopped.

`stopped_ttl` counts from when the instance was first observed stopped. On a
provider where a stopped instance still bills, `validate` warns when `stopped_ttl`
is `null` and `auto_destroy` is `false`: nothing would ever remove the instance.

`gpu sweep` guards every project registered by `gpu setup` with that project's own
`max_lifetime`, lock and `stopped_ttl`. Instances no project tracks are orphans,
judged by the sweep's own `--orphan-max-age` and `--stopped-ttl` flags. See
[Keeping the account clean](USAGE.md#keeping-the-account-clean).

## `health`

| Field | Default | Meaning |
| --- | --- | --- |
| `interval` | `60s` | Probe interval while waiting for the runtime. |
| `boot_timeout` | `30m` | How long the machine may take to reach running, mostly the image pull. Past it the host counts as failed and `setup` tries another (`compute.host_attempts`). |
| `pull_stall_timeout` | `10m` | How long the image pull may go without progress (no new line from the provider, or only retries of a layer) before the host counts as failed and is replaced, without waiting out `boot_timeout`. Healthy pulls measured live showed a new line at least every 8 minutes; stalled ones never recovered. `off` disables it. |
| `startup_timeout` | `10m` | Once the machine runs, how long the model server may take to answer. Raise it for very large models. |
| `path` | `/health` | Health path on the model endpoint. |

A passing health check does not count as activity. A healthy but unused server still goes idle.

## `budget`

| Field | Default | Meaning |
| --- | --- | --- |
| `max_hourly_cost_usd` | – | Refuse to provision above this hourly price. |
| `max_session_cost_usd` | – | The remote guard shuts down once the current session's cost (session age x hourly price) passes this. |

Override once with `gpu setup --allow-cost-override`.

## `workload`

| Field | Default | Meaning |
| --- | --- | --- |
| `mode` | `local` | `local` runs your command on your machine against the remote endpoint; `remote` runs it on the instance over SSH. |
| `env` | `{}` | Extra environment variables for the workload. |
| `workdir` | project dir | Working directory for the command. |

## Safety invariant

An instance may remain billable only while useful work is running, useful work is
queued or imminent within the idle window, or the user explicitly locked it.
Everything else ends in `STOPPED`, or in `DESTROYED` where stopping does not end
billing.
