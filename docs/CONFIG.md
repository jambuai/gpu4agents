# gpu.yaml reference

`gpu.yaml` is the source of truth for the desired state. It describes what should
run and how long it may live. It must never contain credentials, and it never contains
provider API details.

`gpu.yaml` is a manifest shared across Jambu.ai Lab sub-projects. `gpu` reads only
its own `gpu_runtime:` top-level key and ignores every other key in the document, so
other Jambu tools can keep their own config in the same file:

```yaml
# gpu.yaml
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
under the `gpu.yaml` filename, for a solo project that prefers that name.

Resolution order (first match wins, searched from the working directory upward):
`--config PATH`, else `gpu.yaml`, `gpu.yml`, `jambu.yaml`, `jambu.yml`, `config.yml`,
`config.yaml` in that order. `jambu.yaml` is the name used before 0.9.1; projects that
still have one keep working, and renaming it to `gpu.yaml` changes nothing else. The same
goes for `jambu.models.yaml`, now `gpu.models.yaml`. When a directory has both shapes, the shared, namespaced one is preferred over
the legacy flat one.

`${VAR}` references in string values are expanded from the environment.

Unknown keys are a hard error, so a typo in `idle_timeout` cannot silently disable
your cost protection. This applies within the `gpu_runtime:` section. `gpu` does not
check keys that other tools own elsewhere in the document.

### `model_profile`: share model config across files

Set `model_profile: <name>` instead of declaring `model:`/`compute:`/`runtime:`
directly, and gpu merges in that named entry from `gpu.models.yaml`. The catalog is
searched the same way as `gpu.yaml` (this directory, then parents), so one catalog
serves every sub-project in a repo. Fields you declare locally still win over the
profile's. See
[`docs/CHOOSING_A_MODEL.md`](CHOOSING_A_MODEL.md#dont-write-a-new-jambuyaml-per-model---use-the-catalog)
for why this exists and `gpu profiles` / `gpu inspect-model
--add-profile` to manage the catalog.

## `provider`

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | `vast` | Adapter to use: `vast`, `runpod`, `lambda`, or one you added. `gpu providers` lists what is available. `gpu init` writes the one `gpu login` connected. |
| `profile` | `default` | Credential profile. Looks up `VAST_API_KEY__<profile>` (or `RUNPOD_API_KEY__<profile>`, ...) before the plain name. |
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

### `provider.options` for runpod

A pod is a container. Its model and watchdog ports are reached through RunPod's
HTTPS proxy (`https://POD-PORT.proxy.runpod.net`), which ends a single request after
100 seconds; streamed answers are not cut. RunPod's API has no spot pods, and `gpu`
does not SSH into pods, so `workload.mode: remote` is refused (use `gpu job`).

| Field | Default | Meaning |
| --- | --- | --- |
| `api_base` | `https://api.runpod.io` | REST API v2 endpoint. |
| `cloud` | `any` | `secure` (RunPod's own data centers), `community` (vetted hosts) or `any` (the cheaper of the two). |
| `volume_gb` | `0` | Persistent volume at `/workspace`. With none, a stopped pod costs nothing but downloads the model again when it starts. With one, a stopped pod bills the volume, so `lifecycle.stopped_ttl` applies. |
| `skip_low_availability` | `false` | Leave out GPU types RunPod reports as scarce. |

`compute.region` picks data centers by prefix: `US`, `EU-RO-1`, `US-TX,CA`.

### `provider.options` for lambda

A Lambda machine is a whole VM. `gpu` runs the model image on it with Docker, and
opens the model and watchdog ports with a firewall ruleset it creates per region
(`gpu4agents-PORT-PORT`). Lambda cannot stop a machine, only terminate it, so where
another provider would stop, `gpu` and the watchdog terminate. Instance types have a
fixed GPU count (1, 2, 4 or 8) and a fixed disk; `compute.gpu.count` must match one.
`gpu` keeps the SSH key it registers with Lambda in `$GPU_HOME/ssh/lambda_ed25519`.

| Field | Default | Meaning |
| --- | --- | --- |
| `api_base` | `https://cloud.lambda.com/api/v1` | API endpoint. |
| `image_family` | – | VM image, e.g. `lambda-stack-22-04`. Unset: Lambda's default, which has Docker and the NVIDIA container toolkit. |
| `allow_arm64` | `false` | Also rent Arm machines (GH200). The usual model images are x86 only. |
| `source_network` | `0.0.0.0/0` | Who may reach the model and watchdog ports. The model still demands its API key and the watchdog its token. |

`compute.region` picks regions by prefix: `us-east`, `europe-central-1`.

## `compute`

`gpu inspect-model <repo>` reads a model's HuggingFace metadata and suggests
values for this section. [`docs/CHOOSING_A_MODEL.md`](CHOOSING_A_MODEL.md) lists
which source each number comes from.

| Field | Default | Meaning |
| --- | --- | --- |
| `gpu.min_vram_gb` | `24` | Minimum VRAM per GPU. Size for the peak, not the weights. `gpu init` raises it for image models whose peak the weights hide (SDXL 16, FLUX 24), and `validate` warns below that. |
| `gpu.count` | `1` | Number of GPUs on one machine. |
| `gpu.name_filter` | – | Substring match, e.g. `A100`, `RTX 4090`. `gpu setup --offer ID` pins the approved offer's card the same way for one setup, and caps the price at its quote. |
| `gpu.cuda_min` | – | Minimum host CUDA version. `gpu` raises it to what the runtime image needs, read from its tag (`-cu129` means 12.9, vLLM's bare `v0.32+` and `latest` mean 13.0), so an image never lands on a driver too old to run it. |
| `gpu.compute_cap_min` | – | Minimum GPU compute capability (architecture), e.g. `8.0` for Ampere. Distinct from `cuda_min`: a Pascal (GTX 10xx, 6.1) or Volta (V100, 7.0) card can sit behind a modern CUDA-13 driver, and so clear `cuda_min`, yet still be too old for a runtime image whose kernels are built for newer architectures. The image then crashes at engine startup. `gpu` raises this to what the runtime needs: a CUDA-13 vLLM image floors it at `8.0` (those wheels drop Volta), a CUDA-12 vLLM image at `7.0`. The cheapest Vast offers are often these old cards, so without the floor the search keeps picking a GPU the image cannot run. |
| `disk_gb` | `80` | Disk. Must fit the model weights plus the container. |
| `region` | – | Comma-separated geolocations. |
| `spot` | `false` | Use interruptible/bid capacity where supported. |
| `host_attempts` | `3` | How many machines `setup` tries. A host that fails to boot (the provider reports it failed, it vanishes, or it is not running within `health.boot_timeout`) is destroyed, replaced from fresh offers (each held to the budget), and remembered in `$GPU_HOME/bad_hosts.json` so no setup rents it again for 24h. A model server that dies once its container runs fails at once, since the same config would fail on any machine, unless its log shows the host's network failed it (a dropped connection or timeout while downloading weights). The server is then restarted on the same machine, up to 4 times, and the download resumes from what is already on disk; only when the network keeps failing does it count as a host failure. Needs `lifecycle.cleanup_on_setup_failure`; without it the first failed host is kept for inspection. |
| `image` | – | Override the runtime's container image. vLLM's default is `vllm/vllm-openai:v0.32.0`, except on Vast, where it is `vastai/vllm:v0.32.0-cuda-13.0`. Pin a tag: `validate` warns on `latest` or an untagged image, because the CUDA it needs can change under you. The Vast default (the same vLLM, packaged by Vast) is cached on many hosts, where the 12 GB Docker Hub image often stalled past `health.boot_timeout`. |

## `model`

| Field | Default | Meaning |
| --- | --- | --- |
| `id` | *required* | HuggingFace repo id. Optional with `runtime.engine: none` or `custom`. |
| `revision` | – | Pin a commit/branch/tag. |
| `hf_token_env` | `HF_TOKEN` | Env var holding the HF token for gated models. |
| `trust_remote_code` | `false` | Pass `--trust-remote-code` to the runtime. |
| `served_name` | – | Name the model is served under (`--served-model-name`). |
| `task` | `auto` | With `runtime.engine: hf`: `chat`, `embeddings`, `image`, `transcription`, `speech` or `pipeline`, which picks the OpenAI routes ([SERVING.md](../SERVING.md)). `auto` reads the model card's `pipeline_tag` when the server starts; `gpu init` writes the card's answer. |

## `runtime`

How the model runs. Independent from where it runs.

| Field | Default | Meaning |
| --- | --- | --- |
| `engine` | `vllm` | `vllm` (chat models), `hf` (any other Hugging Face model, gpu's own OpenAI-compatible server), `custom` (a server you write, in `service_dir`) or `none` (bare compute for jobs, start nothing). `hf` and `custom` are deployed after boot: changing the model or the server code redeploys on the same machine at the next `gpu setup`. See [SERVING.md](../SERVING.md). |
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
| `packages` | `[]` | `hf`: extra pip packages the model needs, installed before it loads. Pin them. |
| `service_dir` | `server` | `custom`: the directory holding your server and its `gpujob.yaml`, relative to the project. |

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
| `stopped_ttl` | `24h` | Destroy an instance that has been `STOPPED` this long. A stopped Vast instance no longer bills GPU time but still bills storage. `null` keeps stopped instances forever. Enforced only from your computer (`gpu sweep --install-timer`, which `gpu agent install` offers); `gpu stop` says whether that is set up. |
| `auto_stop` | `true` | The guard may stop the instance. |
| `auto_destroy` | `false` | The guard destroys instead of stopping. |
| `keep_disk` | `false` | Where a stopped instance still bills storage (Vast.ai, a RunPod pod with a volume), the guard destroys the instance instead of stopping it, because nothing on a stopped machine can remove it later. `true` stops it, keeping the disk for a restart in about a minute, and leaves the removal to `stopped_ttl`, which only `gpu guard` or `gpu sweep` on your computer enforce (`gpu sweep --install-timer`). |
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

`stopped_ttl` counts from when the instance was first observed stopped. It is
enforced from your computer, by `gpu guard`, `gpu sweep` or any command that
checks the instance. With `keep_disk: true` on a provider where a stopped instance
still bills, `validate` warns when `stopped_ttl` is `null`: nothing would ever
remove the instance.

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

## `queue`

Present means queue mode: a coordinator runs jobs on a pool of machines rented from this same file. Absent, the project behaves exactly as before. The full guide is [QUEUE.md](../QUEUE.md). Nothing here holds an address or a key: `GPU_QUEUE_URL`, `GPU_OUTPUTS_URL` and the `AWS_*` variables come from the environment or `~/.gpu/credentials`, and a URL in this section is a config error.

| Field | Default | Meaning |
| --- | --- | --- |
| `backend` | `local` | `local` (SQLite in the project), `redis` (Redis 5 or newer, one node, not Cluster) or `module:factory` ([QUEUE_BACKENDS.md](QUEUE_BACKENDS.md)). |
| `outputs` | `local` | Where bundles and results live: `local` (`gpu-outputs/`), `s3` or `module:factory`. |
| `namespace` | none | Keyspace name on a shared backend (`[A-Za-z0-9_.-]`, up to 64). Set it for Redis: without it the keyspace follows the config fingerprint and changes when the runtime part of the file changes. `GPU_QUEUE_NAMESPACE` is the fallback. |
| `min_machines` | `0` | Machines kept on with an empty queue. They still time out by `idle_timeout` while the queue is paused. |
| `max_machines` | `1` | The most machines at once. Not below `min_machines`. |
| `max_usd_per_hour` | none | Cap on the whole pool's bill per hour (running machines, stopped ones that still bill their disk, machines held for outputs). It blocks opening machines; it never stops one with a job on it. |
| `scale_up_after` | `30s` | Open a machine when the oldest queued job has waited this long. |
| `jobs_per_machine` | `1` | Jobs one machine runs at once. |
| `max_attempts` | `3` | Runs per job before it is failed with every attempt's reason. Drained attempts do not count. |
| `unresponsive_after` | `5m` | How long a silent machine is waited for before it is destroyed and its jobs requeued. |
| `max_machine_lifetime` | none | Optional hard ceiling on a machine's age. Off: a warm machine is never replaced for being old. |
| `tick` | `5s` | The coordinator's loop period. Leases are 3 ticks (at least 15 s) for the coordinator and 12 ticks (at least 60 s) for a job. |
| `preflight` | built in | `module:function` that returns a `PreflightResult` with the job's limits, or a reason to refuse. |
| `health_check.command` | built in | Replaces the built-in check (GPU visible, a small CUDA operation). It must not write job outputs: they would count as activity and hold the idle shutdown. |
| `health_check.min_free_vram_gb` | none | Free memory threshold. Off by default: a machine serving a model holds most of it. |
| `health_check.timeout` | `2m` | Time allowed for one check. |
| `health_check.between_jobs` | `true` | Check again between jobs. |
| `defaults.timeout` | `1h` | A job's timeout when its `gpujob.yaml` has none. |
| `defaults.max_cost_usd` | none | A job's cost limit when it has none. |
| `defaults.deadline_margin` | `10m` | Added to a job's timeout when the machine's deadline is moved. |

A job's `gpujob.yaml` gains two optional fields: `max_cost_usd` and `min_vram_gb`.

## Safety invariant

An instance may remain billable only while useful work is running, useful work is
queued or imminent within the idle window, or the user explicitly locked it.
Everything else ends in `STOPPED`, or in `DESTROYED` where stopping does not end
billing.
