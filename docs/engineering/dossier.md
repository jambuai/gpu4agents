# gpu4agents: engineering dossier
- **Repository:** https://github.com/jambuai/gpu4agents  **Commit:** <filled in by jambu-research on registration>
- **Scope:** the code on `main` at 8e385f1 (version 0.14.0 plus unreleased work): the provider adapter layer (Vast.ai, RunPod, Lambda and third-party adapters), setup and serving (vLLM, llama.cpp, gpu's own Hugging Face server, custom servers), the on-instance watchdog and the cost caps, jobs and the queue, the MCP server and the agent skill, the CLI, and briefly the local gateway and dashboard. It leaves out the marketing site (`site/`), the product videos, and anything that only a live run could show: no system was run and no provider was called to write this.

## 1. What it is and the problem it solves

A coding agent working without a GPU can neither serve an open-weights model nor run CUDA code. gpu4agents lets the agent rent a cloud GPU on its own, serve a Hugging Face model on it behind an OpenAI-compatible API or run its own code there as a job, bring the results home, and stop paying when the work ends. The hard part is the last one: the agent, the laptop or the network can die at any time, and the machine must still shut down. Most of the engineering below is about that guarantee.

- The package installs one console command under two names, `gpu` and `gpu4agents`, both pointing at `gpu4agents.cli.main:main`. [ref: pyproject.toml#L54-L57]
- Three providers are built in (Vast.ai, RunPod, Lambda), and installed adapters are loaded from the `gpu4agents.providers` entry-point group. [ref: gpu4agents/providers/registry.py#L26-L26] [ref: gpu4agents/providers/registry.py#L126-L149]
- The MCP server registers 23 tools, from `gpu_init` to `gpu_ls`. [ref: gpu4agents/agent/mcp.py#L554-L882]
- An engine is chosen from the model card: chat models (or a card with nothing to go on) go to vLLM, every other task to gpu's own `hf` server. [ref: gpu4agents/serve/__init__.py#L27-L45]
- The lifecycle rule (idle timeout, max lifetime, session budget, locks, workloads) is one pure function, `policy.evaluate`. [ref: gpu4agents/core/policy.py#L206-L278]
- That same `core/policy.py` source is inlined into the script that runs on the rented machine, so the CLI and the machine judge with the same rule. [ref: gpu4agents/watchdog/builder.py#L60-L99]

## 2. Architecture

There are four places where code runs. **The user's computer** runs the CLI (`gpu ...`), the MCP server that an agent talks to, and optionally a sweep timer, a local gateway and a dashboard. **The rented machine** runs the model server (or a job) and the watchdog, a single stdlib-only Python script generated at setup. **The provider's API** is reached only through an adapter. **Optional queue backends** (SQLite or Redis, local directory or S3) hold queue state and outputs.

Everything on the user's side goes through one class, `Engine`, which holds the configuration, the provider adapter, the runtime (engine) and the local state store, and owns setup, reconcile, status, jobs, stop and destroy.

- `Engine` builds its provider through the registry unless one is injected, then takes the runtime's image from the adapter's preferred images. [ref: gpu4agents/core/engine.py#L122-L125]
- The lifecycle policy takes provider behaviour from three capability flags (`stop`, `destroy`, `stop_eliminates_billing`) copied from the adapter into the policy configuration. [ref: gpu4agents/core/engine.py#L142-L156]
- MCP tools do not call the engine in-process: they run the CLI as a subprocess, `<python> -m gpu4agents --json <args>`; `gpu_setup` starts it as a background job, and `gpu_run` runs it without `--json`. [ref: gpu4agents/agent/mcp.py#L204-L226] [ref: gpu4agents/agent/mcp.py#L287-L298] [ref: gpu4agents/agent/mcp.py#L342-L360]
- JSON output is written to stdout as one indented document by `print_json`, and a `JambuError` exits with the exception's `exit_code` (1 when it has none). [ref: gpu4agents/cli/_common.py#L83-L92] [ref: gpu4agents/cli/main.py#L169-L177]
- The watchdog script is `core/policy.py`, then the provider's stop/destroy action source, then `watchdog/agent_body.py`, concatenated into one file. [ref: gpu4agents/watchdog/builder.py#L60-L99]
- An adapter supplies those actions as `WatchdogSpec.actions_source`; the dataclass's comment requires it to define `provider_stop(ctx)` and `provider_destroy(ctx)` with the standard library only, and the watchdog calls those two names. [ref: gpu4agents/providers/base.py#L30-L46] [ref: gpu4agents/watchdog/agent_body.py#L350-L357]
- When the watchdog is enabled, the boot script installs it before it runs the model server's start command. [ref: gpu4agents/watchdog/builder.py#L195-L250]
- The CLI talks to the watchdog over HTTP; when reconcile can reach it, the watchdog's lock, startup flag and last activity are merged into the local state, and its events are pulled into the local log. [ref: gpu4agents/core/engine.py#L432-L455]
- Local state lives per project in `state.json`, next to a `runtime.lock` file and a `logs` directory, all under the project's state directory. [ref: gpu4agents/core/state.py#L326-L334]

## 3. Mechanisms

### Providers: The provider contract and capabilities
- `GPUProvider` is an ABC with eight abstract methods: `validate`, `setup`, `start`, `run`, `status`, `stop`, `destroy`, `logs` [ref: gpu4agents/providers/base.py#L75-L105].
- `find_offers` and `list_instances` are optional: the base class raises `UnsupportedCapabilityError` unless an adapter overrides them [ref: gpu4agents/providers/base.py#L121-L131].
- `watchdog_spec` and `boot_script` default to `None`, and `verify_credentials` defaults to `{}` with no network call [ref: gpu4agents/providers/base.py#L109-L143].
- The default `refresh` re-reads `status()` and falls back to the handle it was given when the status carries no instance [ref: gpu4agents/providers/base.py#L145-L148].
- `ProviderCapabilities` holds 13 boolean flags. `destroy` and `stop_eliminates_billing` default to True and the other 11 default to False [ref: gpu4agents/core/models.py#L62-L86].
- `require_capability(name)` raises `UnsupportedCapabilityError` when that flag is false on the adapter [ref: gpu4agents/providers/base.py#L152-L156].
- `is_configured()` is true only when every name in `required_credentials` resolves, and always true when the adapter needs none [ref: gpu4agents/providers/base.py#L158-L162].
- The engine copies `stop`, `destroy` and `stop_eliminates_billing` from the adapter instance's capabilities into the lifecycle `PolicyConfig` [ref: gpu4agents/core/engine.py#L142-L156].
- Unless `auto_destroy` is set (DESTROY if supported) or `auto_stop` is off (no action), the policy picks STOP when stop is supported and either a stop ends billing or `keep_disk` is set. Otherwise it picks DESTROY when destroy is supported, and STOP only as a last resort when destroy is not [ref: gpu4agents/core/policy.py#L192-L203].
- The engine applies `pull_stall_timeout_s` only when the adapter declares `boot_progress`. Otherwise only `boot_timeout_s` applies [ref: gpu4agents/core/engine.py#L1096-L1100].
- The engine sets the runtime's `provider_image` from the adapter's `preferred_images.get(engine)` [ref: gpu4agents/core/engine.py#L125-L125]. The base default is empty, and of the built-in adapters only Vast declares one, for vLLM [ref: gpu4agents/providers/base.py#L67-L67] [ref: gpu4agents/providers/vast/provider.py#L148-L148].

### Providers: Registry and third-party adapters
- `load_builtin_providers` registers Vast, RunPod and Lambda, then loads installed adapters from the `gpu4agents.providers` entry-point group [ref: gpu4agents/providers/registry.py#L26-L26] [ref: gpu4agents/providers/registry.py#L140-L149].
- An entry point that fails to import, or that does not resolve to a `GPUProvider` subclass, is skipped without an error [ref: gpu4agents/providers/registry.py#L131-L137].
- `register` keys adapters by their `name` class attribute and rejects an empty name or `"abstract"`. Registering the same name again replaces the earlier class [ref: gpu4agents/providers/registry.py#L33-L38].
- When `provider.plugin` is set in the config, `class_for` loads that plugin and requires its `name` to equal `provider.name`, then registers it [ref: gpu4agents/providers/registry.py#L57-L68].
- `load_plugin` accepts `path.py:Class` (resolved against the project directory when the path is relative) or `module:Class`, and raises `ConfigError` unless the target is a `GPUProvider` subclass [ref: gpu4agents/providers/registry.py#L89-L123].
- An unknown provider name raises `ConfigError` that lists the registered names and points to `gpu providers new <name>` [ref: gpu4agents/providers/registry.py#L47-L55].
- The engine builds its provider through `load_builtin_providers()` and `registry.get(config, credentials)` unless a provider is injected [ref: gpu4agents/core/engine.py#L122-L123].
- `scaffold.write_template` writes `gpu_providers/<name>.py` from a template. It refuses names that do not match `[a-z][a-z0-9_-]*`, the built-in names, and an existing file unless `force` is set [ref: gpu4agents/providers/scaffold.py#L241-L262].
- In the template, `find_offers`, `setup`, `start`, `run`, `stop`, `destroy`, `list_instances` and `_to_instance` call `todo()`, which raises `UnsupportedCapabilityError` naming the file [ref: gpu4agents/providers/scaffold.py#L68-L70] [ref: gpu4agents/providers/scaffold.py#L169-L225].

### Providers: Credential resolution
- `CredentialResolver.get` checks the process environment first, then the project `.env`, then `~/.gpu/credentials` (under `gpu_home()`), then any extra sources [ref: gpu4agents/core/credentials.py#L49-L67].
- In each file, a profile-scoped key `NAME__<profile>` wins over plain `NAME`. Environment variables have no profile scoping [ref: gpu4agents/core/credentials.py#L55-L66].
- Each parsed dotenv file is cached by path for the life of the process. `save_user_credential` is the only code that clears that cache, and only for the user credentials file [ref: gpu4agents/core/credentials.py#L17-L35] [ref: gpu4agents/core/credentials.py#L120-L120].
- `save_user_credential` rewrites `~/.gpu/credentials` with file mode 0600 and the directory set to 0700, replacing an earlier line with the same name [ref: gpu4agents/core/credentials.py#L98-L121].
- The registry builds a resolver from the project directory and `provider.profile` when none is passed in [ref: gpu4agents/providers/registry.py#L80-L83].

### Providers: Watchdog action injection
- An adapter supplies `WatchdogSpec(actions_source, env, context)`. The dataclass's comment requires the source to define `provider_stop(ctx)` and `provider_destroy(ctx)` using only the standard library [ref: gpu4agents/providers/base.py#L30-L46].
- When the watchdog is enabled and the adapter returns `None`, `_build_onstart` raises `UnsupportedCapabilityError` [ref: gpu4agents/core/engine.py#L587-L596].
- `build_agent_source` strips docstrings from the action source and inlines it between the core policy and the agent body. With no source, it uses fallback actions that raise [ref: gpu4agents/watchdog/builder.py#L39-L46] [ref: gpu4agents/watchdog/builder.py#L60-L99].
- `WatchdogSpec.env` and `.context` are written into the watchdog config as `env` and `provider_context` [ref: gpu4agents/watchdog/builder.py#L130-L142]. At start the agent calls `os.environ.setdefault` for each `env` entry [ref: gpu4agents/watchdog/agent_body.py#L1038-L1039].
- On the instance, `terminate()` builds `ctx` from `provider_context` plus `instance_id`, then calls `provider_destroy` or `provider_stop`. A failure is counted and returns before `terminated` is set, so only a success marks the machine terminated [ref: gpu4agents/watchdog/agent_body.py#L325-L382].
- `stop_keeps_disk` goes into the watchdog config only when the adapter has both `stop` and `persistent_disk` [ref: gpu4agents/core/engine.py#L616-L617] [ref: gpu4agents/watchdog/builder.py#L143-L146].

### Providers: Vast.ai
- Capabilities: every flag is true except `stop_eliminates_billing`, which is false; the code comment says a stopped instance still bills storage [ref: gpu4agents/providers/vast/provider.py#L127-L144].
- State mapping puts any status message containing "error" or "failed" first (FAILED). Next, intended or current state "stopped" gives STOPPED. Next, a running contract with no `actual_status` gives PROVISIONING [ref: gpu4agents/providers/vast/mapper.py#L34-L53].
- A Vast status of "running" maps to STARTING instead of READY while the intended status is running and the instance has a `start_date` but no public IP [ref: gpu4agents/providers/vast/mapper.py#L55-L62].
- `stopped_hourly_cost_usd` is computed as `storage_cost * disk_space / 730` and is `None` when either field is missing or zero [ref: gpu4agents/providers/vast/mapper.py#L152-L161] [ref: gpu4agents/providers/vast/mapper.py#L186-L186].
- The offer query always filters for rentable, unrented, non-external offers. By default it also requires verified hosts, reliability >= 0.95, download >= 500 Mbps and CUDA >= 12.1 [ref: gpu4agents/providers/vast/mapper.py#L195-L216] [ref: gpu4agents/providers/vast/config.py#L30-L48].
- Excluded hosts are sent to Vast as `machine_id notin` [ref: gpu4agents/providers/vast/mapper.py#L232-L235]. `offer_matches` checks the exclusion again on the client side [ref: gpu4agents/providers/vast/mapper.py#L300-L301].
- `name_filter` is not sent to the server. It is applied on the client side as a case-insensitive substring match [ref: gpu4agents/providers/vast/mapper.py#L223-L229] [ref: gpu4agents/providers/vast/mapper.py#L302-L304].
- When the VRAM-filtered search returns nothing, `find_offers` searches again without the VRAM filter and leaves the VRAM check to `offer_matches` [ref: gpu4agents/providers/vast/provider.py#L271-L281].
- `find_offers` sorts the offer whose id equals `spec.preferred_offer_id` first, then by price [ref: gpu4agents/providers/vast/provider.py#L282-L284].
- `setup` refuses a boot script larger than 16384 bytes before renting anything [ref: gpu4agents/providers/vast/provider.py#L287-L292] [ref: gpu4agents/providers/vast/config.py#L27-L28].
- `setup` tries up to the first five offers and moves to the next one only on `CapacityUnavailableError` or `VastNotFound` [ref: gpu4agents/providers/vast/provider.py#L304-L333].
- The client classifies a `{"success": false}` body before it looks at the status code. Messages about an ask that is gone become `CapacityUnavailableError`, and messages about credit or balance become `ProvisioningError` [ref: gpu4agents/providers/vast/client.py#L120-L146].
- `stop` sends `state: stopped` and returns quietly when the instance is not found or the error says "not running" or "already". `destroy` returns quietly on not-found [ref: gpu4agents/providers/vast/provider.py#L410-L425].
- The Vast watchdog actions call the REST API with `VAST_API_KEY`. They treat a 404 as already gone and get the instance id from `ctx` or from `VAST_CONTAINERLABEL`, `CONTAINER_LABEL` or `CONTAINER_ID` [ref: gpu4agents/providers/vast/provider.py#L62-L109].
- `watchdog_spec` puts the API key and API base into `WatchdogSpec.env`, so they travel in the watchdog config [ref: gpu4agents/providers/vast/provider.py#L464-L470].

### Providers: RunPod
- Capabilities: no SSH, no remote exec and no spot. Stop is supported and by default ends billing [ref: gpu4agents/providers/runpod/provider.py#L129-L142].
- When `options.volume_gb` is set, that adapter instance's capabilities change to `persistent_disk=True` and `stop_eliminates_billing=False` [ref: gpu4agents/providers/runpod/provider.py#L150-L154].
- RunPod statuses map as follows: PROVISIONING and STARTING to themselves, RUNNING to READY, EXITED to STOPPED, ERROR to FAILED, TERMINATED to DESTROYED. Any other status maps to UNKNOWN [ref: gpu4agents/providers/runpod/mapper.py#L17-L38].
- An offer is one synthetic `"<cloud>:<gpuTypeId>"` per GPU type and cloud, priced as per-GPU price × count. AMD cards and GPU types with no price are dropped [ref: gpu4agents/providers/runpod/mapper.py#L58-L89].
- Compute capability is looked up from the GPU id and name in the static table in `providers/gpus.py`. A name the table does not list returns `None`, and the offer is then not filtered on it [ref: gpu4agents/providers/runpod/mapper.py#L87-L87] [ref: gpu4agents/providers/gpus.py#L64-L70] [ref: gpu4agents/providers/runpod/mapper.py#L105-L107].
- `setup` tries offers in price order, up to 6 of them [ref: gpu4agents/providers/runpod/provider.py#L266-L267] [ref: gpu4agents/providers/runpod/provider.py#L272-L272] [ref: gpu4agents/providers/runpod/config.py#L27-L28].
- When a create fails with a generic `ProviderError`, setup looks for a pod with the same name created no earlier than 120 seconds before the create request, and adopts it. It re-raises only if it finds none [ref: gpu4agents/providers/runpod/provider.py#L299-L304] [ref: gpu4agents/providers/runpod/provider.py#L398-L408].
- The boot script travels gzip+base64 in the `JAMBU_BOOT` env var and is unpacked by the container command [ref: gpu4agents/providers/runpod/mapper.py#L26-L34] [ref: gpu4agents/providers/runpod/mapper.py#L170-L170]. The create body is capped at 102400 bytes [ref: gpu4agents/providers/runpod/provider.py#L286-L290].
- The API key reaches the pod as `JAMBU_RUNPOD_KEY` in the pod environment [ref: gpu4agents/providers/runpod/provider.py#L279-L282]. `watchdog_spec` therefore returns an empty `env` [ref: gpu4agents/providers/runpod/provider.py#L391-L394].
- When a stop gets HTTP 409, `stop` succeeds only if the pod is already stopped or gone. Otherwise it raises `ProviderError` [ref: gpu4agents/providers/runpod/provider.py#L357-L370].
- Endpoints are RunPod proxy URLs on port 443 and are filled in only when the pod is READY [ref: gpu4agents/providers/runpod/mapper.py#L125-L147].

### Providers: Lambda
- Capabilities: `stop=False` and `logs=False`. `ssh` and `remote_exec` are true [ref: gpu4agents/providers/lambdacloud/provider.py#L137-L150].
- `start`, `stop` and `logs` raise `UnsupportedCapabilityError` [ref: gpu4agents/providers/lambdacloud/provider.py#L320-L321] [ref: gpu4agents/providers/lambdacloud/provider.py#L373-L380].
- The watchdog's `provider_stop` calls `provider_destroy`, so both terminate the machine through the API [ref: gpu4agents/providers/lambdacloud/provider.py#L115-L121].
- Lambda statuses map as follows: booting to PROVISIONING, active to READY, unhealthy to UNKNOWN, and terminating, terminated or preempted to DESTROYED [ref: gpu4agents/providers/lambdacloud/mapper.py#L19-L34].
- An active machine that has a watchdog endpoint is reported as STARTING until a TCP connect to that port succeeds [ref: gpu4agents/providers/lambdacloud/provider.py#L406-L426].
- Offers are `type@region` pairs from regions with capacity. Non-x86_64 types are skipped unless `allow_arm64` is set, and the GPU count must equal the configured count exactly [ref: gpu4agents/providers/lambdacloud/mapper.py#L64-L93].
- Each launch is tagged with a random nonce. After a failed launch, setup finds the machine by that tag, and the watchdog uses the same tag to find its own instance id when `ctx` carries none [ref: gpu4agents/providers/lambdacloud/provider.py#L263-L305] [ref: gpu4agents/providers/lambdacloud/provider.py#L101-L112].
- The cloud-init user data terminates the machine itself when the image pull fails three times or `docker run` fails [ref: gpu4agents/providers/lambdacloud/mapper.py#L186-L197] [ref: gpu4agents/providers/lambdacloud/mapper.py#L204-L214].
- `setup` creates an ed25519 key under `gpu_home()` the first time and reuses a matching key on the account. It also reuses or creates one firewall ruleset per region and port set [ref: gpu4agents/providers/lambdacloud/provider.py#L439-L471].

### Providers: Shared HTTP client (RunPod, Lambda)
- `JsonClient.send` retries a 429 on any method, and a 502, 503 or 504 only on GET, waiting 2, 4, 8, 15 and 15 seconds or the `Retry-After` value capped at 60 seconds [ref: gpu4agents/providers/http.py#L19-L27] [ref: gpu4agents/providers/http.py#L78-L94].

### Providers: Offer pinning and bad hosts (engine side)
- `_pin` sets `name_filter` to the quoted GPU and caps the price at the quote plus a small rounding tolerance (`_price_tolerance`), or at the configured cap if that is lower. It also sets `preferred_offer_id` [ref: gpu4agents/core/engine.py#L736-L745].
- With a pinned offer, setup calls `find_offers(spec, limit=1)` first and raises `CapacityUnavailableError` without renting when nothing matches [ref: gpu4agents/core/engine.py#L831-L840].
- `BadHosts` keeps failed host ids per provider for 24 hours in `bad_hosts.json`. An unreadable file reads as empty, and a failed write is ignored [ref: gpu4agents/core/hosts.py#L17-L58].
- Setup seeds `spec.exclude_hosts` from `BadHosts.recent`. It makes up to `compute.host_attempts` attempts when `cleanup_on_setup_failure` is set, otherwise one [ref: gpu4agents/core/engine.py#L825-L830].
- When a failed host has been destroyed, its id is added to the exclusions and marked bad [ref: gpu4agents/core/engine.py#L984-L1001]. A failing host that is not discarded for a retry is also marked bad when its host id is known [ref: gpu4agents/core/engine.py#L854-L862].

### Setup and serving: Engine choice from the model card
- `serving_for(card)` maps the HF card's `pipeline_tag` through `PIPELINE_TAG_TASK`, falls back to `library_name` (`diffusers` -> image, `sentence-transformers` -> embeddings), maps any other tag to `pipeline`, and returns `vllm` when the task is `chat` or unknown, else `hf`. [ref: gpu4agents/serve/__init__.py#L27-L45]
- `gpu init` calls `detect_serving` only when `--engine auto` (or `--engine hf` with no `--task`), and `detect_serving` imports and calls `serving_for` on the card from `fetch_hf_api_metadata`. [ref: gpu4agents/cli/setup.py#L131-L135] [ref: gpu4agents/cli/setup.py#L238-L251]
- When no model was given, or `GPU_NO_MODEL_CARD` is set, or the card cannot be read, `detect_serving` returns `vllm` without asking the card for an engine. [ref: gpu4agents/cli/setup.py#L243-L250]
- `KNOWN_VRAM_FLOORS` gives (minimum, safe) GB for repo ids containing `stable-diffusion-xl`/`sdxl` (16, 24), `flux` and `stable-diffusion-3` (24, 32). [ref: gpu4agents/core/model_inspect.py#L374-L388]

### Setup and serving: Runtime registry and container shape
- `build_spec` raises the offer filter's `cuda_min` and `compute_cap_min` to the stricter of the config and what the runtime image needs. [ref: gpu4agents/core/engine.py#L515-L521] [ref: gpu4agents/core/engine.py#L547-L567]
- When the vLLM image's CUDA need is known, vLLM needs compute capability 8.0 if it is >= 13.0, else 7.0; the CUDA need comes from a `-cuNNN` tag suffix or, for bare `vllm/vllm-openai` tags >= v0.32 and `latest`, 13.0. [ref: gpu4agents/runtime/vllm.py#L98-L123] [ref: gpu4agents/runtime/vllm.py#L131-L142]

### Setup and serving: Server command lines
- vLLM runs `python -m vllm.entrypoints.openai.api_server` with model, host 0.0.0.0, port, dtype and gpu-memory-utilization; tensor parallelism defaults to `compute.gpu.count` when above 1; `vllm_args` (flattened) then `extra_args` are appended last. [ref: gpu4agents/runtime/vllm.py#L168-L213] [ref: gpu4agents/runtime/vllm.py#L229-L229]
- vLLM tool calling: `off` passes no flags; otherwise an explicit `tool_call_parser`, else a guess from `model.id`, adds `--enable-auto-tool-choice --tool-call-parser`. [ref: gpu4agents/runtime/vllm.py#L160-L166] [ref: gpu4agents/runtime/vllm.py#L199-L201]
- llama runs `llama-server -hf <model.id>[:<quant>]` on 0.0.0.0 with `--n-gpu-layers` (999 when unset), `--metrics`, `--jinja`, optional `--ctx-size`/`--api-key`, then `llama_args` and `extra_args`. [ref: gpu4agents/runtime/llama.py#L84-L112]
- Both launchers restart the server up to 4 times (delay `15*i+15` s) only when the last 40 new log lines match `HOST_NETWORK_FAILURE_PATTERN`; any other exit ends the launcher with the server's code. [ref: gpu4agents/runtime/vllm.py#L90-L90] [ref: gpu4agents/runtime/vllm.py#L225-L235] [ref: gpu4agents/runtime/llama.py#L55-L55] [ref: gpu4agents/runtime/llama.py#L127-L134] [ref: gpu4agents/runtime/base.py#L22-L27]

### Setup and serving: Boot script composition (model-server part)
- `build_onstart_script` writes the runtime env as `export` lines and also appends each pair to `/etc/jambu/runtime.env`, installs the watchdog (if enabled), and only then runs the runtime start command. [ref: gpu4agents/watchdog/builder.py#L191-L193] [ref: gpu4agents/watchdog/builder.py#L195-L249]
- `_build_onstart` passes the runtime's start command, env, pid path, log path and activity probe (the probe only when `traffic_counts_as_activity`) into the boot script and watchdog config. [ref: gpu4agents/core/engine.py#L598-L630]

### Setup and serving: Endpoint and per-machine API key
- `_api_key` returns `runtime.api_key` when set, else a fresh `secrets.token_urlsafe(32)`; it returns None only for a runtime with no `api_key_env` (`none`). [ref: gpu4agents/core/engine.py#L575-L585]
- A generated key is exported as the runtime's `api_key_env` (`VLLM_API_KEY`, `LLAMA_API_KEY`, `JAMBU_API_KEY`); a configured key is passed as `--api-key` instead. [ref: gpu4agents/core/engine.py#L619-L623] [ref: gpu4agents/runtime/vllm.py#L193-L194] [ref: gpu4agents/runtime/llama.py#L107-L108] [ref: gpu4agents/runtime/service.py#L63-L63]
- `_provision` stores the key in state when it creates an instance; the restart path in `ensure_ready` does not regenerate it. [ref: gpu4agents/core/engine.py#L903-L908] [ref: gpu4agents/core/engine.py#L1035-L1046]
- `connection()` returns, when an endpoint exists, `base_url = <endpoint>/v1` and the key (state's, else configured, else `jambu-local`), plus `model = served_name or model.id`, engine, task and routes. [ref: gpu4agents/core/engine.py#L1286-L1309]

### Setup and serving: Setup, reuse and restart
- Under the process lock, `setup` reconciles; with an instance whose fingerprint differs it raises `ConfigError` unless `--recreate`; with a match it reuses the instance (via `ensure_ready` when waiting); with `--recreate` it destroys the old one first. [ref: gpu4agents/core/engine.py#L785-L801]
- `ensure_ready` starts a STOPPED instance (only if the provider can stop/start), waits for it to run, connects the watchdog, deploys the service for `hf`/`custom`, and waits for health unless engine is `none` (then it calls `mark_ready`). [ref: gpu4agents/core/engine.py#L1035-L1067]
- `_wait_running` polls provider status every 6 s until the instance is running with its named endpoints published (`model` unless engine is `none`, `watchdog` when enabled), up to `boot_timeout`; vanish, FAILED or timeout raise `HostFailure`. [ref: gpu4agents/core/engine.py#L1106-L1156] [ref: gpu4agents/core/engine.py#L1158-L1164]
- A pull with no new progress line for `pull_stall_timeout` raises `HostFailure`, but only on providers whose `boot_progress` capability is set; a line matching `retrying` is not progress. [ref: gpu4agents/core/engine.py#L1097-L1099] [ref: gpu4agents/core/engine.py#L1135-L1146] [ref: gpu4agents/core/engine.py#L90-L100]
- `_wait_runtime` polls `endpoint + health.path` with `wait_for_healthy` up to `startup_timeout`; a probe passes on any status below 400. [ref: gpu4agents/core/engine.py#L1201-L1252] [ref: gpu4agents/core/health.py#L17-L24]
- Defaults: `boot_timeout` 30m, `pull_stall_timeout` 10m, `startup_timeout` 10m, `health.interval` 60s, `health.path` `/health`; the `hf` and `custom` templates set `startup_timeout: 30m`. [ref: gpu4agents/core/config.py#L276-L286] [ref: gpu4agents/core/config.py#L926-L928] [ref: gpu4agents/core/config.py#L975-L976]
- The health wait aborts early when the provider says the instance is gone, FAILED or STOPPED, or the watchdog reports `runtime_alive: false`; the latter error carries the last 30 lines of the runtime log. [ref: gpu4agents/core/engine.py#L1221-L1244] [ref: gpu4agents/core/engine.py#L1263-L1280] [ref: gpu4agents/core/engine.py#L80-L80]

### Setup and serving: Setup failure and cleanup
- `setup` tries `compute.host_attempts` (default 3) hosts only when `cleanup_on_setup_failure` is true, else 1. [ref: gpu4agents/core/engine.py#L825-L825] [ref: gpu4agents/core/config.py#L116-L116]
- Between attempts the failed host is destroyed and excluded; if that destroy fails, setup stops instead of renting a second machine. [ref: gpu4agents/core/engine.py#L854-L858] [ref: gpu4agents/core/engine.py#L972-L1001]
- On any `JambuError` or `KeyboardInterrupt` during provisioning, `_fail_setup` emits `setup_failed` and, when `cleanup_on_setup_failure` and an instance id exist, destroys the instance; a failed destroy is logged, not raised. [ref: gpu4agents/core/engine.py#L874-L878] [ref: gpu4agents/core/engine.py#L1003-L1016]

### Setup and serving: Pinned offer and rented vs quoted
- With `--offer`, the quote comes from the saved `offers.json` (or a fresh search when the adapter has `pricing_query`); `_pin` sets the GPU name filter, caps price at the quote plus a small rounding tolerance (`_price_tolerance`) and prefers that offer id. [ref: gpu4agents/core/engine.py#L718-L745]
- When there is a quote and a rented GPU name, `rented_vs_quoted` sets `matches_quote: false` with a warning if the quoted GPU name is not a substring of the rented one or the rented price exceeds quote + tolerance; the connection printer adds the comparison to the JSON payload, or prints the warning in text mode. [ref: gpu4agents/core/engine.py#L747-L770] [ref: gpu4agents/cli/setup.py#L357-L363]

### Setup and serving: Fingerprint
- The fingerprint is the first 16 hex chars of SHA-256 over provider name, GPU block, disk, image, model id/revision/trust_remote_code/served_name and runtime fields (engine, port, api_key, dtype, context length, tool-calling settings, `vllm_args`, `extra_args` and others). [ref: gpu4agents/core/config.py#L602-L636]
- For `hf`/`custom` the model block is dropped and runtime is reduced to engine, port and api_key. [ref: gpu4agents/core/config.py#L628-L634]

### Setup and serving: Service engines (`hf`, `custom`) and the custom contract
- `_deploy_service` requires the watchdog, adds `JAMBU_API_KEY` to the service env, and leaves a running service with the same digest alone; otherwise it cancels running services and submits the new one as a job labelled `service`. [ref: gpu4agents/core/engine.py#L1545-L1584]
- `custom` reads `<service_dir>/gpujob.yaml` (default `server`, relative to the project); a missing directory or a manifest without `run:` is a `ConfigError`. [ref: gpu4agents/runtime/service.py#L157-L177]
- The template server answers `/health` 503 until `load()` finishes, serves `/metrics` without auth, requires `Bearer $JAMBU_API_KEY` on other routes (none when the key is empty), and exits 1 if `load()` raises. [ref: gpu4agents/serve/template/server.py#L26-L26] [ref: gpu4agents/serve/template/server.py#L68-L86] [ref: gpu4agents/serve/template/server.py#L121-L132]

### Setup and serving: gpu's own HF server
- `hf` uploads `serve/hf`, runs `bash install.sh` then `server.py`, with `JAMBU_TASK`, `JAMBU_DTYPE`, `JAMBU_PACKAGES`, and optional revision / trust-remote-code env. [ref: gpu4agents/runtime/service.py#L108-L128]
- The model loads in a background thread; `/health` is 503 until ready; a load failure exits the process with code 1. [ref: gpu4agents/serve/hf/server.py#L968-L977] [ref: gpu4agents/serve/hf/server.py#L699-L704]
- POST routes need the Bearer key when one is set, are limited to the task's routes, and run one at a time under a GPU lock; bodies over 200 MB get 413. [ref: gpu4agents/serve/hf/server.py#L732-L775] [ref: gpu4agents/serve/hf/server.py#L45-L45] [ref: gpu4agents/serve/hf/server.py#L588-L588] [ref: gpu4agents/serve/hf/server.py#L646-L660]

### Setup and serving: gpu.yaml schema
- `model.id` is required unless engine is `none` or `custom`. [ref: gpu4agents/core/config.py#L512-L519]

### Watchdog and cost caps: The lifecycle rule (core/policy.py, shared by CLI and watchdog)

- `evaluate` checks rules in a fixed order: a STOPPED instance can only get the `stopped_ttl` destroy (else `already_terminal`); DESTROYED is terminal; then the queue deadline; then `max_lifetime`; then the session budget; then the lock; then startup, pending workloads, and finally `idle_timeout`. [ref: gpu4agents/core/policy.py#L206-L278]
- `stopped_ttl` destroys a STOPPED instance only when `stopped_ttl_s` is set, `stopped_since` is known, the stop is older than the TTL, the provider supports destroy, and no lock is active. [ref: gpu4agents/core/policy.py#L220-L231]
- The queue deadline is checked before the lock and before pending workloads, so it ends a machine under work and over a lock, but only when `deadline_enforced` is true, a deadline is set and a terminal action is available. [ref: gpu4agents/core/policy.py#L235-L240]
- `max_lifetime` is measured from `session_started_at` (falling back to `created_at`), not from creation; an active lock suppresses it. [ref: gpu4agents/core/policy.py#L155-L157] [ref: gpu4agents/core/policy.py#L242-L250]
- The session budget rule applies only when both `max_session_cost_usd` and a truthy `hourly_cost_usd` are set; spend is `hourly_cost_usd * session_age / 3600`, and a lock suppresses it. [ref: gpu4agents/core/policy.py#L252-L261]
- Idle time is zero while any non-stale workload is pending or startup is in progress; otherwise it is `now` minus the latest of `last_activity` and all workload heartbeats. [ref: gpu4agents/core/policy.py#L143-L150]
- A running workload goes stale once its heartbeat is older than `heartbeat_grace_s`; a queued workload never goes stale. [ref: gpu4agents/core/policy.py#L55-L61]
- A lock with no `locked_until` holds only when `allow_indefinite_lock` is true. [ref: gpu4agents/core/policy.py#L117-L123]
- `_terminal_action` picks destroy under `auto_destroy` (nothing if destroy is unsupported); nothing when `auto_stop` is off; stop when the provider can stop and a stop ends billing (or `keep_disk` is set); otherwise destroy if supported, else stop if supported, else nothing. [ref: gpu4agents/core/policy.py#L192-L203]

### Watchdog and cost caps: Building and delivering the watchdog (watchdog/builder.py)

- The remote script is the header, then `core/policy.py`, then the provider's action source (or fallbacks that raise), then `agent_body.py`, each with `from __future__` lines removed and docstrings stripped through `ast` parse/unparse. [ref: gpu4agents/watchdog/builder.py#L60-L99] [ref: gpu4agents/watchdog/builder.py#L39-L46]
- When `agent_body.py` is imported directly rather than inlined, it imports the policy names from the package; the import runs only if `evaluate` is not already in its globals. [ref: gpu4agents/watchdog/agent_body.py#L38-L49]
- The engine builds the config with `policy=self.policy_config()` (no hourly price) and a startup grace of `boot_timeout + startup_timeout + 300s`. [ref: gpu4agents/core/engine.py#L597-L618]
- The onstart script writes the config and agent as zlib+base85 blobs, decodes them under umask 077, then sets the config to mode 600 and the agent to 700. It exports `JAMBU_SESSION_STARTED_AT=$(date +%s)` and starts a `nohup` bash loop that restarts the agent 5 s (`SUPERVISOR_RESPAWN_DELAY_S`) after it exits and forwards TERM to it. All of this comes before the runtime start command. [ref: gpu4agents/watchdog/builder.py#L29-L29] [ref: gpu4agents/watchdog/builder.py#L195-L250]
- If no python interpreter is in the image, the script logs that the watchdog is NOT installed and continues to start the runtime. [ref: gpu4agents/watchdog/builder.py#L222-L225]
- The watchdog token is `secrets.token_urlsafe(32)`, generated fresh on each provision. [ref: gpu4agents/watchdog/builder.py#L49-L51] [ref: gpu4agents/core/engine.py#L804-L806]

### Watchdog and cost caps: The guard loop on the instance (watchdog/agent_body.py)

- `guard_loop` ticks every `interval_s` (default 60) and catches every exception from a tick, so only process exit ends it. [ref: gpu4agents/watchdog/agent_body.py#L488-L506]
- Each tick merges `/identify` policy overrides into the boot policy, clears an expired lock, evaluates the policy, and then runs `_choose` and any `TERMINATE_HOOKS`. A hook may change or veto the choice, and a hook exception is logged and ignored. [ref: gpu4agents/watchdog/agent_body.py#L232-L235] [ref: gpu4agents/watchdog/agent_body.py#L447-L472]
- A pending `/shutdown` request takes priority over the policy decision; when there is no shutdown to do, the stop/action failure counters are reset. [ref: gpu4agents/watchdog/agent_body.py#L400-L416]
- Stop-to-destroy escalation happens only when the chosen action is stop, stops have been failing continuously since `stop_failing_since` for at least `escalate_after_s` (default 1800 s), and destroy is supported; the reason becomes `stop_failed_escalation` with the original reason kept. [ref: gpu4agents/watchdog/agent_body.py#L88-L88] [ref: gpu4agents/watchdog/agent_body.py#L417-L427] [ref: gpu4agents/watchdog/agent_body.py#L474-L477]
- `terminate` is non-blocking on an action lock (a concurrent call is skipped). A provider exception increments `action_failures` (and, for a stop, `stop_failures`) and leaves `terminated` false. Success sets `terminated`, resets the counters, and clears `pending_shutdown`. [ref: gpu4agents/watchdog/agent_body.py#L331-L384]
- After a successful terminate, if the machine is still running once `reissue_after_s` has passed (default `max(300, 5*interval)`, or `reissue_after_s` from the config), the guard clears `terminated` and decides again (`guard_reissue`). [ref: gpu4agents/watchdog/agent_body.py#L387-L391] [ref: gpu4agents/watchdog/agent_body.py#L433-L445]
- Startup grace ends early when the runtime pid is dead, or when `startup_grace_s` (default 900) has passed since the session start. [ref: gpu4agents/watchdog/agent_body.py#L251-L266]
- A new `JAMBU_SESSION_STARTED_AT` value starts a new session: `terminated`, failure counters, `pending_shutdown` and `deadline_at` are reset, startup is re-armed, and running or queued workloads are marked failed. The same value (a supervisor respawn) keeps the state. [ref: gpu4agents/watchdog/agent_body.py#L161-L173] [ref: gpu4agents/watchdog/agent_body.py#L176-L210]

### Watchdog and cost caps: Inference traffic as activity (Prometheus probe)

- The probe scrapes every `min(interval, 15s)`. Traffic means a configured gauge above zero, or a counter higher than at the previous scrape. On the first scrape counters only set the baseline (a gauge above zero still counts), and a counter that drops does not count. [ref: gpu4agents/watchdog/agent_body.py#L569-L583] [ref: gpu4agents/watchdog/agent_body.py#L629-L638]
- A scrape error is never activity; it is logged at most once every `PROBE_ERROR_LOG_EVERY_S` (600 s). [ref: gpu4agents/watchdog/agent_body.py#L593-L600] [ref: gpu4agents/watchdog/agent_body.py#L89-L89]
- The first successful scrape while `starting` ends startup and resets `last_activity`. [ref: gpu4agents/watchdog/agent_body.py#L614-L623]
- For vLLM the probe reads loopback `/metrics`: gauges `vllm:num_requests_running` and `vllm:num_requests_waiting`, counters `vllm:request_success_total` and `vllm:prompt_tokens_total`. It is passed only when `lifecycle.traffic_counts_as_activity` is set. [ref: gpu4agents/runtime/vllm.py#L261-L269] [ref: gpu4agents/core/engine.py#L598-L600]

### Watchdog and cost caps: HTTP control surface and auth

- Every route except `GET /health` requires `X-Jambu-Token`, compared with `hmac.compare_digest`. When the config has no token, every request is authorized. [ref: gpu4agents/watchdog/agent_body.py#L722-L730] [ref: gpu4agents/watchdog/agent_body.py#L757-L763] [ref: gpu4agents/watchdog/agent_body.py#L791-L807]
- `/identify` stores the instance id, saves `hourly_cost_usd` and any `policy` dict as overrides, and marks activity. The CLI calls it once the watchdog answers a ping. [ref: gpu4agents/watchdog/agent_body.py#L869-L883] [ref: gpu4agents/core/engine.py#L1175-L1178]
- `/shutdown` accepts only `stop` or `destroy`, saves the request as `pending_shutdown` so the guard retries it, and runs `terminate(force=True)` in a thread, replying 202. [ref: gpu4agents/watchdog/agent_body.py#L1003-L1018]
- `/lock` refuses an indefinite lock unless `allow_indefinite_lock`; `/deadline` takes only a finite positive time and never moves the deadline backwards. [ref: gpu4agents/watchdog/agent_body.py#L957-L989]
- `/extend` decompresses the source, saves it under `EXT_DIR`, and `exec`s it in the module globals. Saved extensions are loaded again at start, and a failing one is logged and skipped. [ref: gpu4agents/watchdog/agent_body.py#L670-L699] [ref: gpu4agents/watchdog/agent_body.py#L1041-L1043]
- The CLI client turns 401, any status of 400 or above, and transport errors into `WatchdogUnavailable`; the `try_*` calls return None instead. [ref: gpu4agents/core/beacon.py#L35-L62]

### Watchdog and cost caps: CLI side: reconcile, billable time, guard

- Reconcile trusts the provider first. A running provider state goes through `derive_state`. The watchdog's `terminated` flag counts only when the provider says UNKNOWN. A STOPPED or FAILED machine fails its running or queued workloads. [ref: gpu4agents/core/engine.py#L459-L474]
- Billable time accumulates only across billable-compute states (`billed_seconds` plus the open `running_since` period), and `session_cost_usd` is computed from that total. [ref: gpu4agents/core/state.py#L172-L195] [ref: gpu4agents/core/engine.py#L476-L493]
- `stopped_since` keeps the earliest stop, but restarts at `now` when any evidence (session start, last activity, watchdog contact, provider created_at) is more than 1 s newer. Seeing the machine billable again after a stop starts a new session. [ref: gpu4agents/core/state.py#L197-L220] [ref: gpu4agents/core/engine.py#L480-L485]
- The CLI drops `idle_timeout` (and records `idle_skipped: watchdog_unreachable`) when the watchdog is enabled and installed, this reconcile could not read it, and the instance is in a billable state. [ref: gpu4agents/core/engine.py#L178-L192] [ref: gpu4agents/core/engine.py#L2308-L2322]
- `guard_tick` reconciles under the process lock, clears an expired lock and evaluates. Only when `apply` is true does it let `_protect_job_outputs` adjust the decision and terminate through the provider. [ref: gpu4agents/core/engine.py#L2296-L2335]
- `max_hourly_cost_usd` is checked before provisioning against the cheapest matching offer: `BudgetExceededError` unless `--allow-cost-override`, and only a warning when the provider cannot query pricing. [ref: gpu4agents/core/engine.py#L632-L662] [ref: gpu4agents/core/engine.py#L841-L842]

### Watchdog and cost caps: Vanished machines and STALE status

- `_last_seen` is the latest of `last_seen`, watchdog last contact, `last_activity`, session start and creation. [ref: gpu4agents/core/engine.py#L289-L300]
- `_why_missing` replays `evaluate` forward from last seen in steps of `max(60s, span/20000)`. It returns the first rule that calls for a destroy, or None if the first rule due is a stop, if nothing comes due, if the watchdog is disabled or was neither installed nor given a token, or if the machine was already stopped. [ref: gpu4agents/core/engine.py#L302-L339]
- When the provider says the instance does not exist, billing closes at the inferred due time (or last seen), and the worst case goes in `cost_if_ran_until_now_usd`. `instance_destroyed` is emitted with `reason`/`by` `unknown` plus `likely_*` and `inferred`, and only when a rule was found. [ref: gpu4agents/core/engine.py#L370-L414] [ref: gpu4agents/core/state.py#L233-L244]
- If `provider.status` raises, reconcile keeps local state and records `provider_error`. `status` then reports `reconciled: false`, a `stale_reason` and `last_seen`. [ref: gpu4agents/core/engine.py#L360-L367] [ref: gpu4agents/core/engine.py#L2007-L2015]

### Watchdog and cost caps: Account sweep and timer

- `gpu sweep` calls `guard_projects`, which runs each registered project's own `guard_tick` and collects the instance ids those projects track, including from pool slot files and from projects whose config fails to load. [ref: gpu4agents/core/sweep.py#L322-L371] [ref: gpu4agents/cli/account.py#L286-L290]
- Pool slots are guarded with `apply` off while a coordinator holds the lease. When the queue is unreachable, a slot is acted on only if it is past its deadline or its `stopped_ttl`. [ref: gpu4agents/core/sweep.py#L261-L302]
- Orphans (instances with a jambu label that no project tracks): a stopped orphan is destroyed once it has been stopped longer than `stopped_ttl`, counted from the sweep's own first sighting. A running orphan older than `orphan_max_age` is stopped (no action if the provider cannot stop), never destroyed. A running orphan with no creation time is skipped. [ref: gpu4agents/core/sweep.py#L479-L515] [ref: gpu4agents/core/sweep.py#L439-L443]
- The timer is a systemd user oneshot that runs `python -m gpu4agents sweep --apply` with `TimeoutStartSec=10min` and no `-` prefix. It fires 60 s after activation, then every `max(60, every)` s; on macOS a LaunchAgent is used instead. [ref: gpu4agents/cli/timer.py#L57-L94] [ref: gpu4agents/cli/timer.py#L128-L161] [ref: gpu4agents/cli/timer.py#L211-L211]

### Jobs and queue: The gpujob.yaml contract (direct `gpu job run`)

- `gpujob.yaml` is parsed into `JobManifest`, which forbids unknown keys and accepts `run`, `setup`, `env`, `timeout`, `exclude`, `label`, `max_cost_usd` and `min_vram_gb`; a list given for `run`/`setup` is shell-quoted and joined into one command, `env` values are stringified, and `timeout` must parse as a duration. [ref: gpu4agents/core/job.py#L75-L111]
- In `job_submit`, the command-line command, timeout, setup and label win over the manifest, CLI `--env` overrides manifest `env` key by key, and CLI `--exclude` patterns are added to the manifest's (not replacing them). A job with no command from either source raises `ConfigError`. [ref: gpu4agents/core/engine.py#L1439-L1467]
- `gpu job run` refuses a timeout longer than `lifecycle.max_lifetime`; it does not compare the timeout to the session budget. [ref: gpu4agents/core/engine.py#L1446-L1454]
- The bundle always leaves out VCS dirs, `.gpu`, virtualenvs, caches, `node_modules`, `.env`/`.env.*` and `gpu-outputs`, skips symlinks and non-regular files, and raises `ConfigError` when the tar.gz exceeds 2 GiB. [ref: gpu4agents/core/job.py#L46-L68] [ref: gpu4agents/core/job.py#L147-L163] [ref: gpu4agents/core/job.py#L173-L179]
- Besides its own env, a job gets `JAMBU_PROVIDER` and `JAMBU_INSTANCE_ID`, plus `OPENAI_BASE_URL` (loopback), `OPENAI_API_KEY` and `JAMBU_MODEL_ID` only when the project's engine is not `none` and the runtime exposes ports. [ref: gpu4agents/core/engine.py#L1409-L1421] [ref: gpu4agents/core/engine.py#L1465-L1465]

### Jobs and queue: The job runner on the machine (watchdog extension)

- The CLI uploads `watchdog/jobs_ext.py` through `POST /extend` (zlib + base64) only when the watchdog's reported digest of the `jobs` extension differs from the local file's sha256 prefix; a watchdog that reports no `extensions` key raises `StateError` asking for `gpu setup --recreate`. [ref: gpu4agents/core/job.py#L268-L283] [ref: gpu4agents/core/job.py#L216-L217] [ref: gpu4agents/watchdog/agent_body.py#L670-L673] [ref: gpu4agents/watchdog/agent_body.py#L691-L699]
- The runner strips the watchdog's provider credential env keys from the job's environment, defaults `HF_HOME` to `/workspace/hf`, and sets `JAMBU_JOB_ID`, `JAMBU_JOB_DIR`, `JAMBU_OUTPUT_DIR` and `PYTHONUNBUFFERED`; the caller's env is applied last. [ref: gpu4agents/watchdog/jobs_ext.py#L321-L334]
- The job runs as `bash -c` in `src/`, runs `setup.sh` first when there is one and `command.sh` only if setup succeeded, writes its own exit code to `rc`, and is started in a new session with stdout and stderr in one `job.log`. [ref: gpu4agents/watchdog/jobs_ext.py#L336-L351]
- A non-service job is registered as a watchdog workload, and its monitor thread refreshes the workload heartbeat and `last_activity` every `JOB_HEARTBEAT_S` (10 s default) while it runs; a service job (a model server) is not a workload and does not refresh activity. [ref: gpu4agents/watchdog/jobs_ext.py#L375-L385] [ref: gpu4agents/watchdog/jobs_ext.py#L469-L479]
- A job past `timeout_s` is cancelled with reason `timeout`; cancel sends SIGTERM to the process group and SIGKILL after `JOB_CANCEL_GRACE_S` (10 s default) if it is still recorded running. [ref: gpu4agents/watchdog/jobs_ext.py#L465-L468] [ref: gpu4agents/watchdog/jobs_ext.py#L482-L509]
- The final state is the cancel reason when one was set (`cancelled` or `timeout`), `lost` when no exit code is known, else `succeeded` for 0 and `failed` otherwise. [ref: gpu4agents/watchdog/jobs_ext.py#L429-L436]
- When the extension is (re)loaded, a running job whose `session_started_at` differs from the current session and that has no live process handle is finished immediately (from the `rc` file if present); others get a new monitor thread. [ref: gpu4agents/watchdog/jobs_ext.py#L674-L685]
- `gpu job run` exits 0 for `succeeded`, 124 for `timeout`, 130 for `cancelled`, else the job's return code, or 1 when there is none. [ref: gpu4agents/cli/job.py#L27-L29] [ref: gpu4agents/cli/job.py#L50-L59]

### Jobs and queue: How direct-job outputs come home and are protected

- `GET /jobs/<id>/outputs` streams `out/` as a tar.gz; only when the job had already ended does it then record the output directory's (count, bytes, newest mtime) signature as `outputs_saved`. [ref: gpu4agents/watchdog/jobs_ext.py#L562-L584]
- A job's outputs count as unsaved when it is not a service, `out/` is non-empty, and it is still running or its saved signature differs from the current one. [ref: gpu4agents/watchdog/jobs_ext.py#L189-L198]
- The watchdog runs every registered terminate hook on the guard's chosen action; a hook may replace it or return None (no shutdown), and an exception in a hook is logged and does not block the shutdown. [ref: gpu4agents/watchdog/agent_body.py#L463-L472] [ref: gpu4agents/watchdog/jobs_ext.py#L691-L691]
- `outputs_guard` leaves explicit `/shutdown` requests and `stop_failed_escalation` alone; otherwise, with unsaved outputs, a destroy becomes a stop when the provider supports stop and the watchdog config says a stop keeps the disk. This applies to any policy reason, including `deadline`. [ref: gpu4agents/watchdog/jobs_ext.py#L201-L214] [ref: gpu4agents/core/engine.py#L616-L616]
- Where a stop would not keep the disk, only an `idle_timeout` shutdown is vetoed, for up to `OUTPUTS_HOLD_S` (7200 s default, env-overridable) after the last unsaved job ended; other reasons proceed and log `outputs_lost`. [ref: gpu4agents/watchdog/jobs_ext.py#L94-L94] [ref: gpu4agents/watchdog/jobs_ext.py#L215-L222]
- The CLI guard (`guard_tick` with apply) passes its decision through `_protect_job_outputs`, which downloads finished jobs' outputs while the machine is up, and before an automatic destroy downloads unsaved outputs, turning the destroy into a stop (provider with stop and persistent disk) or no action (idle reason, or machine not running) when some cannot be fetched. [ref: gpu4agents/core/engine.py#L2323-L2324] [ref: gpu4agents/core/engine.py#L1868-L1906]
- `gpu destroy` downloads unsaved outputs first and raises `StateError` instead of destroying when some remain, unless `discard_outputs` is set. [ref: gpu4agents/core/engine.py#L2238-L2247]

### Jobs and queue: Queue data model and backend contract

- Queue job states are queued, claimed, running, delivering, succeeded, failed, cancelled; the last three are final. Attempt outcomes include `drained` and `not_started`. [ref: gpu4agents/queue/models.py#L12-L38]
- The backend is chosen by name: `local` (SQLite) and `redis` load built-in modules, and `module:factory` loads a project factory. [ref: gpu4agents/queue/backend.py#L100-L120]
- The local backend stores jobs in SQLite (WAL) under `<state_dir>/queue/queue.db`; every write transaction is `BEGIN IMMEDIATE` and first reaps expired leases back to QUEUED, or to CANCELLED when cancel was requested. [ref: gpu4agents/queue/backends/local.py#L48-L50] [ref: gpu4agents/queue/backends/local.py#L64-L75] [ref: gpu4agents/queue/backends/local.py#L97-L107] [ref: gpu4agents/queue/backends/local.py#L335-L336]
- Writes are fenced by token: `_owned` raises `LeaseLost` unless the stored token matches the claim's and the lease has not expired. [ref: gpu4agents/queue/backends/local.py#L120-L124]
- `submit` is idempotent on `key` and stamps `submitted_at` with the backend clock; `claim` returns nothing while paused unless a specific job id is asked for, and skips jobs with `cancel_requested`. [ref: gpu4agents/queue/backends/local.py#L137-L166]
- The Redis backend runs every state change as a Lua script that reads the server's `TIME` and reaps expired leases first; its URL is the `GPU_QUEUE_URL` credential read through `CredentialResolver` (a `ConfigError` when it is missing), and without `queue.namespace` or `GPU_QUEUE_NAMESPACE` its namespace is derived from the config fingerprint. [ref: gpu4agents/queue/backends/redis.py#L44-L49] [ref: gpu4agents/queue/backends/redis.py#L70-L90] [ref: gpu4agents/queue/backends/redis.py#L483-L500]

### Jobs and queue: Submit, cancel, retry, pull (api.py)

- `submit_job` requires a non-empty `run:` in `gpujob.yaml`, returns the existing job for a known key without storing anything, uploads the bundle to the output store, and deletes the bundle only if the backend write failed and the job is known absent, or another submit won the key. [ref: gpu4agents/queue/api.py#L116-L126] [ref: gpu4agents/queue/api.py#L141-L179] [ref: gpu4agents/queue/api.py#L182-L188]
- `describe` (used by status, cancel and retry output) replaces every manifest env value with `***`. [ref: gpu4agents/queue/api.py#L223-L235] [ref: gpu4agents/queue/api.py#L290-L290] [ref: gpu4agents/queue/api.py#L303-L303]
- A cancel on a QUEUED job makes it CANCELLED at once; on a leased job it only sets `cancel_requested`. [ref: gpu4agents/queue/backends/local.py#L213-L226]
- `retry` moves only a FAILED job back to QUEUED and keeps its attempts. [ref: gpu4agents/queue/backends/local.py#L228-L241]
- `api.outputs` verifies the stored outputs in place when the target is the store location (the local store's default), else copies them through `fetch_previous`, which in the local store verifies first. [ref: gpu4agents/queue/api.py#L306-L330] [ref: gpu4agents/queue/outputs.py#L258-L261] [ref: gpu4agents/queue/outputs.py#L405-L410]

### Jobs and queue: Coordinator: leadership and standby

- The coordinator lease is max(3 ticks, 15 s), a job lease is max(12 ticks, 60 s), and a heartbeat thread renews both every min(lease)/3. [ref: gpu4agents/queue/coordinator.py#L78-L82] [ref: gpu4agents/queue/coordinator.py#L295-L298] [ref: gpu4agents/queue/coordinator.py#L341-L351]
- A tick that does not hold or win the lease only writes metrics (standby); winning the lease runs `recover()`. [ref: gpu4agents/queue/coordinator.py#L374-L379] [ref: gpu4agents/queue/coordinator.py#L437-L450]
- A coordinator whose renewal is refused is demoted at once: it drops every run from memory without writing anything for them. [ref: gpu4agents/queue/coordinator.py#L452-L463] [ref: gpu4agents/queue/coordinator.py#L465-L482]
- Acquire succeeds only when no live lease exists or the caller already owns it; renew requires a live lease held by the caller (both backends). [ref: gpu4agents/queue/backends/local.py#L285-L299] [ref: gpu4agents/queue/backends/redis.py#L236-L247]
- `recover()` lists every non-final job and marks for adoption those whose `limits` record an in-flight attempt (`remote_id` with `attempt == len(attempts)+1`) or an unfinished settle, then runs the auditor once (when one is configured) and retries the adoptions. [ref: gpu4agents/queue/coordinator.py#L614-L644]

### Jobs and queue: Coordinator: dispatch, deadlines and attempts

- Each attempt's remote id is `<job id>-a<n>`; one longer than 64 characters is rejected (settled FAILED as `preflight_rejected`). [ref: gpu4agents/queue/coordinator.py#L94-L94] [ref: gpu4agents/queue/coordinator.py#L1448-L1455] [ref: gpu4agents/queue/coordinator.py#L672-L673] [ref: gpu4agents/queue/coordinator.py#L706-L712]
- Before a dispatch the coordinator runs preflight (when the run has no limits yet), pushes the machine's deadline to now + timeout + margin (never earlier than the current one; no dispatch if that push fails), and records attempt, remote id, slot, instance id, owner, started_at and deadline_at in the job's `limits`. [ref: gpu4agents/queue/coordinator.py#L1600-L1624] [ref: gpu4agents/queue/coordinator.py#L1652-L1660]
- The coordinator re-checks leadership and job ownership between the upload and the start (`before_start`), and cancels a run it started if it lost the job while dispatching. [ref: gpu4agents/queue/coordinator.py#L1664-L1671] [ref: gpu4agents/queue/pool.py#L574-L580] [ref: gpu4agents/queue/coordinator.py#L1695-L1703]
- A 409 on upload or start is reported as `refused` when the runner's record says `refused`, else as `already_running` (never a second start). [ref: gpu4agents/queue/pool.py#L581-L590]
- The deadline reaches the machine as `POST /deadline`, where the watchdog keeps max(old, new) and rejects non-finite or non-positive values. [ref: gpu4agents/queue/pool.py#L458-L467] [ref: gpu4agents/core/engine.py#L2022-L2031] [ref: gpu4agents/core/beacon.py#L129-L130] [ref: gpu4agents/watchdog/agent_body.py#L976-L989]
- The deadline rule is enforced only on pool-slot engines in queue mode (`deadline_enforced=pooled`), where `max_lifetime` comes from `queue.max_machine_lifetime` and the session budget is off; past the deadline it returns the terminal action even with work running and over a lock. [ref: gpu4agents/core/engine.py#L145-L160] [ref: gpu4agents/core/policy.py#L235-L240]
- While a finished run's outputs wait for delivery, the coordinator pushes the machine's deadline again whenever less than half the margin remains. [ref: gpu4agents/queue/coordinator.py#L1179-L1192]
- A non-zero exit on a machine whose health check passes, and a remote `timeout`, settle FAILED directly (no retry); only machine-side faults go through `_fault`, which requeues until `max_attempts` counted attempts, where `drained` and `not_started` do not count but more than 2x `max_attempts` `not_started` attempts also give up. [ref: gpu4agents/queue/coordinator.py#L89-L89] [ref: gpu4agents/queue/coordinator.py#L104-L104] [ref: gpu4agents/queue/coordinator.py#L1303-L1317] [ref: gpu4agents/queue/coordinator.py#L1408-L1435]
- A retried attempt gets the previous attempt's verified outputs packed into the bundle under `.jambu-previous/`, and its `run` is prefixed with `export JAMBU_PREVIOUS_OUTPUT_DIR=...`; if they cannot be fetched the raw bundle is used. [ref: gpu4agents/queue/coordinator.py#L1640-L1649] [ref: gpu4agents/queue/coordinator.py#L1725-L1746]
- A settle first writes the attempt, outputs ref and a `settling` record with a random settle id under the claim, then calls the backend settle; a job found with a `settling` record whose id differs from `result.settle_id` is finished by the next claimer instead of run again, unless it was a success whose outputs are now bad, in which case it is reopened. [ref: gpu4agents/queue/coordinator.py#L1457-L1475] [ref: gpu4agents/queue/coordinator.py#L680-L686]

### Jobs and queue: Takeover of runs that never started (fencing)

- On takeover, a run the home machine does not list is judged `never` started only when its runner reports its uploads and none is fresh, the attempt was recorded at least 60 s ago (`UPLOAD_SLACK_S`) [ref: gpu4agents/queue/coordinator.py#L86-L86], and the runner's jobs directory is older than the attempt; otherwise `wait` or `unknown`. [ref: gpu4agents/queue/coordinator.py#L153-L181]
- A `never` verdict first fences the remote id (`POST /jobs/<id>/refuse`); only an acknowledged fence requeues the job as `NOT_STARTED`, and a fence answer showing the run did start adopts it. [ref: gpu4agents/queue/coordinator.py#L802-L813] [ref: gpu4agents/queue/coordinator.py#L831-L847]
- The runner records a refused id, deletes its bundle and code, and from then on answers its upload and start with 409. [ref: gpu4agents/watchdog/jobs_ext.py#L280-L295] [ref: gpu4agents/watchdog/jobs_ext.py#L235-L237] [ref: gpu4agents/watchdog/jobs_ext.py#L302-L304]
- A run on a machine this coordinator cannot see is requeued only after the recorded deadline plus `DEADLINE_GRACE_S` (300 s) when a deadline was recorded (otherwise started_at + timeout + margin plus the grace); while undecided, its known home machine gets a keepalive. [ref: gpu4agents/queue/coordinator.py#L184-L200] [ref: gpu4agents/queue/coordinator.py#L866-L879] [ref: gpu4agents/queue/coordinator.py#L815-L822]

### Jobs and queue: Queue outputs: delivery and the sha256 manifest

- Delivery is fetch, health check (on an answering machine: always after a non-zero exit, and between jobs when `health_check.between_jobs`), store, verify, settle; machine-side failures retire the machine after 5, store failures back off exponentially (capped at 300 s) and never retire. [ref: gpu4agents/queue/coordinator.py#L90-L91] [ref: gpu4agents/queue/coordinator.py#L1240-L1301] [ref: gpu4agents/queue/coordinator.py#L1319-L1351]
- A failed health check after a non-zero, non-timeout exit is recorded as UNHEALTHY (retried) and the machine is retired; after a clean exit or a timeout the machine is retired but the job is still delivered. [ref: gpu4agents/queue/coordinator.py#L1270-L1281]
- `put_outputs` hashes every regular file (no symlinks) into a manifest `{path, bytes, sha256}`, stages a copy plus `MANIFEST.json`, and swaps it in with `os.replace`; a failed attempt goes under `.attempt-<n>`, and those dirs survive a later success. [ref: gpu4agents/queue/outputs.py#L55-L64] [ref: gpu4agents/queue/outputs.py#L200-L226]
- `verify` checks the stored manifest equals the recorded file list, each file's size and sha256, and reports any unexpected file outside `.attempt-*`. [ref: gpu4agents/queue/outputs.py#L228-L256]
- Retiring a machine pulls every unsaved queue run's outputs first; a run is requeued only after the destroy succeeded, and a delivering run whose outputs were pulled is delivered from them when it exited 0 or was already health-checked. [ref: gpu4agents/queue/pool.py#L653-L699] [ref: gpu4agents/queue/coordinator.py#L1104-L1143]

### Jobs and queue: Spend limits

- Default preflight rejects a job whose `max_cost_usd` (or the queue default) is below hourly rate x timeout, whose `min_vram_gb` exceeds the pool card, or with no positive timeout. [ref: gpu4agents/queue/preflight.py#L56-L91]
- The scaler decides how many machines to open (plus which idle ready ones to keep): it caps by `max_machines` and by how many offers fit in `max_usd_per_hour` minus the current pool rate, and reports which cap blocked. [ref: gpu4agents/queue/scaler.py#L48-L85]
- Pool rate counts full price for every state except stopped (disk price) and gone/retired (zero). [ref: gpu4agents/queue/spend.py#L22-L31]
- Each billed interval is appended to a JSONL ledger and split equally among the jobs on the machine; a job's cost is the sum over its attempts' remote ids. [ref: gpu4agents/queue/spend.py#L47-L57] [ref: gpu4agents/queue/pool.py#L878-L889] [ref: gpu4agents/queue/coordinator.py#L1501-L1510]

### Jobs and queue: Audit invariants I1..I8 (from code)

- With repair on, the auditor may change jobs only through a leading coordinator; it may cancel or retire on machines only as the leading coordinator, or standalone while no coordinator holds the lease, rechecked before each action. [ref: gpu4agents/queue/audit.py#L169-L171] [ref: gpu4agents/queue/audit.py#L213-L225]
- I1: a RUNNING in-flight job that `find_run`/`lost_detail` say runs nowhere; repair closes the attempt as MACHINE_LOST (requeued, or FAILED at the attempt cap). [ref: gpu4agents/queue/audit.py#L325-L337] [ref: gpu4agents/queue/coordinator.py#L942-L957]
- I2: a remote run in state running, named `<id>-a<n>`, that the job's in-flight record does not place on that machine; repair cancels it. [ref: gpu4agents/queue/audit.py#L261-L309]
- I3: a pending success settle whose outputs fail verification (repair: redeliver), or a settled success failing verification (reported only; up to 100 verified per audit). [ref: gpu4agents/queue/audit.py#L72-L72] [ref: gpu4agents/queue/audit.py#L365-L403]
- I4: a provider instance labeled as this pool's that no slot tracks; retired (when the auditor may touch machines) unless a non-final job's limits name that instance and its deadline plus grace has not passed. [ref: gpu4agents/queue/audit.py#L407-L445]
- I5: pool hourly rate over `max_usd_per_hour` (alert only). [ref: gpu4agents/queue/audit.py#L449-L458]
- I6: a RUNNING job past `deadline_at` still listed as running (or not seen); repair (only when the machine lists it as running) cancels it and closes the attempt as TIMEOUT (requeued, or FAILED at the attempt cap). [ref: gpu4agents/queue/audit.py#L338-L352] [ref: gpu4agents/queue/coordinator.py#L959-L989]
- I7: no lease holder while jobs are not final, or the holder's last renewal older than max(3 ticks, 3 heartbeats); an auditing coordinator skips it when the lease is free or its own. [ref: gpu4agents/queue/audit.py#L70-L70] [ref: gpu4agents/queue/audit.py#L236-L257]
- I8: a `work/<job>/out-a<n>` directory older than one hour with no `.stored` marker [ref: gpu4agents/queue/audit.py#L71-L71], skipping jobs the auditing coordinator is delivering (alert only). [ref: gpu4agents/queue/audit.py#L462-L487]
- The coordinator runs the auditor with repair every 12 ticks; `gpu queue audit` exits 1 while any violation is unrepaired. [ref: gpu4agents/queue/coordinator.py#L397-L402] [ref: gpu4agents/cli/queue.py#L301-L302]

### Agent surfaces and CLI: MCP server: transport and dispatch

- `serve()` reads newline-delimited JSON-RPC from stdin, answers a JSON array (batch) with the list of non-null replies, writes one JSON line per response, and points `sys.stdout` at `sys.stderr` for the duration so stray prints cannot reach the protocol stream. [ref: gpu4agents/agent/mcp.py#L932-L958]
- `initialize` echoes the client's `protocolVersion` only if it is in `PROTOCOL_VERSIONS`, otherwise answers with the first (newest) entry, and returns the `INSTRUCTIONS` text as `instructions`. [ref: gpu4agents/agent/mcp.py#L37-L37] [ref: gpu4agents/agent/mcp.py#L908-L916]
- `call_tool` turns a `ToolError` into a result with `isError: true` carrying the error payload, and turns any other exception into `isError: true` with `exit_code: 1` instead of letting it end the server. [ref: gpu4agents/agent/mcp.py#L886-L898]
- The server registers 23 tools: gpu_init, gpu_validate, gpu_offers, gpu_setup, gpu_status, gpu_connection, gpu_run, gpu_job_run, gpu_job_status, gpu_job_pull, gpu_job_cancel, gpu_logs, gpu_lock, gpu_unlock, gpu_stop, gpu_destroy, gpu_queue_submit, gpu_queue_status, gpu_queue_cancel, gpu_queue_pull, gpu_queue_control, gpu_adopt, gpu_ls. [ref: gpu4agents/agent/mcp.py#L554-L882]

### Agent surfaces and CLI: MCP server: how a tool reaches the engine

- Tools do not call the engine in-process: `cli()` runs `<python> -m gpu4agents --json <args>` as a subprocess with `cwd` set to the project directory, stdin from /dev/null, and a 900 s timeout; a timeout becomes a `ToolError` with `exit_code: 8`. [ref: gpu4agents/agent/mcp.py#L43-L43] [ref: gpu4agents/agent/mcp.py#L204-L226]
- A non-zero CLI exit becomes a `ToolError` whose `error` is the text after `error:` on the last stderr line that has one, plus `exit_code`, and a `meaning`/`next` pair looked up in `EXIT_MEANINGS` by exit code. [ref: gpu4agents/agent/mcp.py#L46-L59] [ref: gpu4agents/agent/mcp.py#L120-L137] [ref: gpu4agents/agent/mcp.py#L228-L231]
- The default project is `--project`, else `$GPU_PROJECT_DIR`, else the server's cwd; a tool's `project_dir` overrides it, and the directory holding the nearest config file (`gpu.yaml`) upwards is used. With no config found the tool fails with `exit_code: 2` and a `next` pointing at gpu_init. [ref: gpu4agents/cli/agent.py#L20-L31] [ref: gpu4agents/agent/mcp.py#L178-L180] [ref: gpu4agents/agent/mcp.py#L187-L202] [ref: gpu4agents/core/config.py#L639-L647]
- `gpu_init` creates the directory, runs `gpu init --model ... [--engine] [--task] [--force]` there, then edits `max_hourly_cost_usd` and `min_vram_gb` in the written gpu.yaml by regex substitution, and for `engine: custom` adds `server_dir` to the result. [ref: gpu4agents/agent/mcp.py#L240-L275]

### Agent surfaces and CLI: MCP server: long setup as a background job

- `gpu_setup` starts `gpu setup [--recreate] [--offer ID]` as a background job only when no setup job for that project is running, then waits `min(wait_seconds, 300)` (default 45 s); while the job runs it returns `status: "running"`, `started_now`, and a `next` telling the agent to call gpu_setup again. [ref: gpu4agents/agent/mcp.py#L40-L41] [ref: gpu4agents/agent/mcp.py#L287-L307]
- When the job succeeded, gpu_setup returns `status: "ready"` with the CLI's connection fields minus `watchdog`; if `matches_quote` is `false` the `next` field becomes the CLI's warning plus "Report the change to the user before starting work". A failed or `lost` job raises a `ToolError` built from its exit code and stderr tail. [ref: gpu4agents/agent/mcp.py#L308-L328]
- `Job.start` launches `python -m gpu4agents.agent.jobs DIR NAME -- argv` in a new session (POSIX) or detached process group (Windows) with all stdio on /dev/null, and records pid, argv and start time in `.gpu/jobs/<name>.json`. [ref: gpu4agents/agent/jobs.py#L30-L30] [ref: gpu4agents/agent/jobs.py#L57-L63] [ref: gpu4agents/agent/jobs.py#L86-L109] [ref: gpu4agents/core/paths.py#L13-L13] [ref: gpu4agents/core/paths.py#L28-L30]
- `gpu_stop` and `gpu_destroy` first cancel a running setup job: `Job.cancel` sends SIGINT to the job's process group and waits up to 120 s; if it has not ended the tool fails with `exit_code: 12`, otherwise the result carries `setup_cancelled`. [ref: gpu4agents/agent/jobs.py#L120-L137] [ref: gpu4agents/agent/mcp.py#L403-L446]

### Agent surfaces and CLI: Confirm gates and offer pinning

- `gpu_destroy` refuses with `exit_code: 2` unless `confirm` is exactly `true`; when confirmed it runs `gpu destroy --yes --reason <reason|agent>` and adds `--discard-outputs` only when `discard_outputs` is set. [ref: gpu4agents/agent/mcp.py#L429-L446] Test: [ref: tests/test_agent.py#L138-L142]
- `gpu_queue_control` accepts only pause/resume/drain, rejects `now` with any action but drain, and refuses `drain` with `now=true` unless `confirm` is `true`. [ref: gpu4agents/agent/mcp.py#L526-L543]
- `offer_id` flows from gpu_setup to `gpu setup --offer` to `engine.setup(offer_id=...)`; the engine looks the id up in the offers file the last `gpu offers` saved, then (only if the provider has `pricing_query`) in a fresh provider search, and otherwise raises a ConfigError. [ref: gpu4agents/agent/mcp.py#L294-L295] [ref: gpu4agents/cli/setup.py#L325-L344] [ref: gpu4agents/core/engine.py#L710-L716] [ref: gpu4agents/core/engine.py#L718-L735] [ref: gpu4agents/core/engine.py#L782-L783]
- A pinned quote sets the GPU name filter, caps the hourly price at the quote plus a small rounding tolerance (`_price_tolerance`) (or the configured cap, if lower), and sets `preferred_offer_id`. [ref: gpu4agents/core/engine.py#L736-L745]
- `gpu setup`'s output merges `rented`, `quoted` and `matches_quote` and drops the watchdog token; `matches_quote` is false when the rented GPU name does not contain the quoted one or the price exceeds quote plus tolerance, and is true when there was no quote or the rented GPU name is unknown. [ref: gpu4agents/cli/setup.py#L352-L358] [ref: gpu4agents/core/engine.py#L747-L770]

### Agent surfaces and CLI: MCP gpu_run and jobs

- `gpu_run` runs `gpu run --label <label|mcp> -- <command>` without `--json`, in its own session on POSIX; past `timeout_seconds` (default 600) it sends SIGINT to the group (terminate on Windows), waits 15 s, then kills; it returns `exit_code`, `timed_out` and stdout/stderr tails of at most 12,000 characters as a normal (non-error) result. [ref: gpu4agents/agent/mcp.py#L42-L44] [ref: gpu4agents/agent/mcp.py#L342-L384]
- `gpu_job_run` always passes `job run --detach --no-start`, so it never provisions; on exit code 3 it replaces `next` with "start the machine first with gpu_setup (it costs money: ask the user)". [ref: gpu4agents/agent/mcp.py#L448-L467] Test: [ref: tests/test_agent.py#L126-L132]

### Agent surfaces and CLI: Agent install (`gpu agent install`)

- The Claude target writes the skill to `.claude/skills/gpu4agents/SKILL.md`; codex, gemini and cursor share one copy under `.agents/skills/`; the base is the home directory with `--global`, else the project. [ref: gpu4agents/agent/install.py#L217-L235]
- MCP registration per target: project `.mcp.json` for Claude (or, with `--global`, `claude mcp remove` then `claude mcp add --scope user`), `[mcp_servers.gpu]` in Codex's config.toml, `mcpServers.gpu` in `.gemini/settings.json`, `.cursor/mcp.json` with `--project ${workspaceFolder}`, and Claude Desktop's config pinned with `--project <abs path>` plus a skill zip in `$GPU_HOME`. [ref: gpu4agents/agent/install.py#L237-L290]
- JSON configs are merged: only `mcpServers.gpu` is written, and an invalid JSON file or a non-object `mcpServers` raises a ConfigError instead of being overwritten. [ref: gpu4agents/agent/install.py#L98-L119]
- The installer copies `gpu4agents/agent/skill/SKILL.md`; a test asserts the published `skills/gpu4agents/SKILL.md` has the same text. [ref: gpu4agents/agent/install.py#L46-L51] [ref: tests/test_agent.py#L437-L439]
- `gpu agent install` asks about the sweep timer only when neither flag is given, `--json` is off and stdin is a TTY; installing the timer requires the provider credentials to be in a file, not only the shell environment. [ref: gpu4agents/cli/agent.py#L69-L75] [ref: gpu4agents/agent/install.py#L303-L327] [ref: gpu4agents/cli/account.py#L186-L211]

### Agent surfaces and CLI: CLI plumbing

- The root callback sets `--json`, `--config`, `--verbose`, and for every subcommand except mcp, ls, sweep, gateway, dashboard and queue runs the stopped-and-billing warning check. [ref: gpu4agents/cli/main.py#L23-L23] [ref: gpu4agents/cli/main.py#L49-L69]
- `--json` output is one indented JSON document on stdout; errors print `error: ...` on stderr and exit with the exception's `exit_code`; `main()` also maps a stray `JambuError` to its code and Ctrl-C to 130. [ref: gpu4agents/cli/_common.py#L83-L92] [ref: gpu4agents/cli/main.py#L169-L177]
- `gpu init` refuses (exit 2) to write a project in a temp or scratchpad folder unless `--allow-temp`, and, unless `--flat`, also scaffolds `scripts/`, `work/` and a `CLAUDE.md` only where they do not exist. [ref: gpu4agents/core/paths.py#L45-L72] [ref: gpu4agents/cli/setup.py#L93-L105] [ref: gpu4agents/cli/setup.py#L220-L221] [ref: gpu4agents/cli/setup.py#L286-L298]
- `gpu login` without `--check` needs a TTY (hidden prompt) or `--stdin`, verifies the values with the provider adapter before saving, writes each to `~/.gpu/credentials` (file mode 0600, directory 0700), and records `GPU_PROVIDER` when `--provider` was given or a provider was picked at the prompt. [ref: gpu4agents/cli/account.py#L26-L26] [ref: gpu4agents/cli/account.py#L418-L451] [ref: gpu4agents/core/credentials.py#L98-L121]

### Agent surfaces and CLI: gpu claude

- `claude_env` points `ANTHROPIC_BASE_URL` at the gateway's listen address, sets the model and all Opus/Sonnet/Haiku/subagent aliases to the served model, and sets max context to `runtime.context_length` (default 131072) and max output to `max(4096, min(32000, ctx // 8))`. [ref: gpu4agents/cli/claude.py#L31-L53]
- `gpu claude` starts `gpu gateway start` in a new session if a HEAD to the gateway does not return 200 (polling 20 x 0.5 s), exits 127 if `claude` is not on PATH, then `execve`s Claude Code with that env. [ref: gpu4agents/cli/claude.py#L67-L115]

### Agent surfaces and CLI: Gateway

- Each request gets up to two attempts, the second with a forced endpoint refresh after a connect error; with no endpoint it answers 503; responses are re-sent chunked; HEAD answers 200 as a liveness probe. [ref: gpu4agents/gateway/server.py#L75-L123]
- Built-in adapters: `none` passes through; `openai` strips `Authorization`; `anthropic` also strips `x-api-key` and `anthropic-*` headers containing "key" or "auth"; both injecting adapters add the instance key as Bearer and `x-api-key` and rename `model` to the served one when enabled. [ref: gpu4agents/gateway/adapters.py#L51-L108]
- `anthropic-openai` converts an Anthropic Messages body to OpenAI chat, rewrites `/v1/messages` to `/v1/chat/completions` keeping any query string, and translates the response back (SSE when the request had `stream`, an Anthropic error envelope for status >= 400 or a 200 body with `error` and no `choices`). [ref: gpu4agents/gateway/anthropic_openai.py#L232-L303]

### Agent surfaces and CLI: Dashboard

- `DashboardServer` refuses to start (ValueError) on a non-loopback host with no token when `token_required_off_loopback` is set; `gpu dashboard install` generates a token file `.gpu/dashboard.token` (0600) for a non-loopback bind that has none. [ref: gpu4agents/dashboard/server.py#L23-L39] [ref: gpu4agents/dashboard/server.py#L191-L192] [ref: gpu4agents/cli/dashboard.py#L86-L87] [ref: gpu4agents/cli/dashboard.py#L180-L186]
- POST `/api/control/<action>` returns 404 unless `control_enabled`, then requires the token gate, a JSON body with `confirm: true` (else 400), and, when a project index is set (control mode), a project this host knows (else 409). [ref: gpu4agents/dashboard/server.py#L144-L179]
- With control enabled, actions shell out to the `gpu` CLI in the project's directory (`setup --offer`, `stop`, `destroy --yes`, `lock`) or run `systemctl --user restart` on the project's gateway unit. [ref: gpu4agents/cli/dashboard.py#L101-L130] [ref: gpu4agents/dashboard/ops.py#L24-L31] [ref: gpu4agents/dashboard/ops.py#L68-L93]

## 4. Design decisions

### Providers

- **The core branches on capabilities, not on provider names.** The engine reads flags such as `stop_eliminates_billing` and `boot_progress` from the adapter [ref: gpu4agents/core/engine.py#L142-L156] [ref: gpu4agents/core/engine.py#L1096-L1099]. A test enforces it by searching `core/` for `== "vast"` [ref: tests/test_engine.py#L307-L317].
- **Stop vs destroy is a policy decision driven by capabilities.** A stop that still bills escalates to destroy unless `keep_disk` is set [ref: gpu4agents/core/policy.py#L192-L203]. Lambda gets the same result in another way: its watchdog stop action terminates the machine [ref: gpu4agents/providers/lambdacloud/provider.py#L119-L120].
- **Watchdog credentials travel two ways.** Vast puts the key in `WatchdogSpec.env`, which is written into the watchdog config [ref: gpu4agents/providers/vast/provider.py#L464-L470] [ref: gpu4agents/watchdog/builder.py#L139-L139]. RunPod and Lambda put it in the container environment at create time and return an empty `env` [ref: gpu4agents/providers/runpod/provider.py#L279-L282] [ref: gpu4agents/providers/runpod/provider.py#L391-L394] [ref: gpu4agents/providers/lambdacloud/provider.py#L265-L271] [ref: gpu4agents/providers/lambdacloud/provider.py#L395-L398]. Trade-off: the Vast path puts the key into the watchdog config.
- **Adopting a create whose answer was lost.** RunPod matches by name and creation time, and Lambda by nonce tag, so a machine that was created but never acknowledged is still tracked [ref: gpu4agents/providers/runpod/provider.py#L299-L304] [ref: gpu4agents/providers/runpod/provider.py#L398-L408] [ref: gpu4agents/providers/lambdacloud/provider.py#L300-L306]. Vast has no such path: only capacity and not-found errors from `create_instance` move on to the next offer, and any other `ProviderError` propagates [ref: gpu4agents/providers/vast/provider.py#L309-L313].
- **Vast has its own HTTP client.** It repeats the retry logic of the shared `JsonClient` instead of subclassing it, and only RunPod and Lambda subclass `JsonClient` [ref: gpu4agents/providers/vast/client.py#L30-L101] [ref: gpu4agents/providers/http.py#L78-L90] [ref: gpu4agents/providers/runpod/client.py#L22-L34] [ref: gpu4agents/providers/lambdacloud/client.py#L26-L27].
- **Vast stops search fallbacks at the first well-formed answer.** `search_offers` tries three endpoint shapes and stops at the first one that answers with an `offers` key, even when the list is empty [ref: gpu4agents/providers/vast/client.py#L168-L194].
- **Compute capability comes from a static name table when the API lacks it.** Vast reports compute capability ×100 [ref: gpu4agents/providers/vast/mapper.py#L85-L88]. RunPod and Lambda look it up from the name table [ref: gpu4agents/providers/gpus.py#L14-L70] [ref: gpu4agents/providers/runpod/mapper.py#L87-L87] [ref: gpu4agents/providers/lambdacloud/mapper.py#L83-L83]. Trade-off: an unlisted card is not filtered on architecture [ref: gpu4agents/providers/runpod/mapper.py#L105-L106].

### Setup and serving

- hf/custom servers start after boot as watchdog service jobs and are left out of the machine fingerprint, so a model or code change redeploys on the same machine. vLLM's command is baked into the boot script, so a model change needs `--recreate`. The cost: service engines refuse to run without the watchdog. [ref: gpu4agents/core/config.py#L628-L634] [ref: gpu4agents/core/engine.py#L1594-L1600] [ref: gpu4agents/core/engine.py#L1548-L1552]
- The endpoint gets a key by default (a generated per-machine key, not an open port). The cost: the key sits in plain text in the boot script and `/etc/jambu/runtime.env` on the instance, and `recover_boot_secrets` reads it back for `gpu adopt`. [ref: gpu4agents/core/engine.py#L575-L585] [ref: gpu4agents/core/engine.py#L619-L628] [ref: gpu4agents/watchdog/builder.py#L191-L193] [ref: gpu4agents/watchdog/builder.py#L279-L287] [ref: gpu4agents/core/engine.py#L2138-L2144]
- A network failure while downloading weights restarts the server in place (bounded at 4) before the host is blamed. Only failures that match the network pattern are retried; anything else fails at once. [ref: gpu4agents/runtime/vllm.py#L90-L90] [ref: gpu4agents/runtime/vllm.py#L225-L235] [ref: commit:f15adab]
- Default images are pinned: vLLM by tag, llama by digest. `validate` warns on `latest` or untagged images because the CUDA requirement can change under an unpinned tag. [ref: gpu4agents/runtime/vllm.py#L95-L95] [ref: gpu4agents/runtime/llama.py#L41-L44] [ref: gpu4agents/core/engine.py#L258-L264] [ref: gpu4agents/runtime/base.py#L63-L67]
- GPU architecture and driver floors come from the image, so offers that cannot run it are filtered out before renting. The cost: offers are fewer, e.g. Volta is excluded for CUDA 13 vLLM. [ref: gpu4agents/core/engine.py#L547-L567] [ref: gpu4agents/runtime/vllm.py#L134-L142] [ref: commit:f372274]
- Retrying another host requires `cleanup_on_setup_failure`. With cleanup off, the first failed host is kept and no second machine is rented. [ref: gpu4agents/core/engine.py#L821-L825] [ref: gpu4agents/core/engine.py#L1003-L1010]

### Watchdog and cost caps

- **One policy source for both guards.** `policy.py`'s source is inlined into the remote script (with docstrings and comments stripped), so the watchdog and the CLI run the same `evaluate`. Cost: `policy.py` must stay stdlib-only, as its module docstring requires. [ref: gpu4agents/watchdog/builder.py#L60-L99] [ref: gpu4agents/core/policy.py#L1-L8] [ref: gpu4agents/core/engine.py#L332-L332] [ref: gpu4agents/watchdog/agent_body.py#L455-L455]
- **Destroy instead of stop where a stop still bills.** Before commit 4c1658e, `_terminal_action` returned STOP even when a stop did not end billing. Now it destroys unless `keep_disk` is set. Cost: the disk is lost by default on such providers. [ref: commit:4c1658e] [ref: gpu4agents/core/policy.py#L198-L203] [ref: tests/test_policy.py#L235-L252]
- **Escalation waits for continuous failure.** Escalation used to happen after 3 failed stops (commit 8589df9). It now waits for 30 minutes of continuous failure, and the counters reset when the machine is back in use (commit 77ea1e4). This trades up to 30 more minutes of billing for not losing the disk to a short API outage. [ref: commit:8589df9] [ref: commit:77ea1e4] [ref: gpu4agents/watchdog/agent_body.py#L85-L88] [ref: gpu4agents/watchdog/agent_body.py#L410-L427]
- **Running orphans are only ever stopped.** The sweep never destroys a running untracked instance (stop, or nothing if the provider cannot stop). A stopped orphan is destroyed once `stopped_ttl` has passed. [ref: gpu4agents/core/sweep.py#L488-L499] [ref: gpu4agents/core/sweep.py#L509-L515] [ref: tests/test_sweep.py#L149-L154]
- **The session budget is not enforced in pool mode.** For a pooled slot `max_session_cost_usd` is None and `max_lifetime` comes from `queue.max_machine_lifetime_s`; the deadline is enforced instead. [ref: gpu4agents/core/engine.py#L145-L160]
- **Hourly price reaches the watchdog late.** The boot config has no hourly price, so the remote session budget is inactive until `/identify` supplies `hourly_cost_usd`. [ref: gpu4agents/core/engine.py#L610-L610] [ref: gpu4agents/core/policy.py#L253-L253] [ref: gpu4agents/watchdog/agent_body.py#L875-L877]
- **Reported cost and capped cost measure different spans.** `session_cost_usd` on the CLI side is computed from all billable seconds, and `mark_session_start` does not reset `billed_seconds`. The budget rule counts only the current session's age. After a restart, the reported `session_cost_usd` can be above what the cap measures. [ref: gpu4agents/core/engine.py#L490-L493] [ref: gpu4agents/core/state.py#L218-L220] [ref: gpu4agents/core/policy.py#L254-L254]

### Jobs and queue

- Jobs run inside the watchdog process as an uploaded extension rather than a separate agent: the CLI pushes the runner source by digest, so a runner change reaches existing machines without re-provisioning; cost: a watchdog without `/extend` cannot run jobs and must be recreated. [ref: gpu4agents/core/job.py#L268-L283]
- The job's exit code is written by its own shell to a file so a respawned watchdog still learns how a job ended. [ref: gpu4agents/watchdog/jobs_ext.py#L335-L341] [ref: gpu4agents/watchdog/jobs_ext.py#L408-L420]
- Results-safety is enforced twice, on the machine (terminate hook) and in the CLI guard, so it holds with no laptop connected and also when the CLI decides. Trade-off: on stop-keeps-disk providers an unsaved result turns a destroy into a stop, which keeps billing for disk where a stop still bills. [ref: gpu4agents/watchdog/jobs_ext.py#L210-L214] [ref: gpu4agents/core/engine.py#L1894-L1906]
- Only machine-side failures are retried; a job that exits non-zero is the job's fault and is settled FAILED, with `gpu queue retry` as the manual path (attempts kept, so earlier counted attempts still count toward `max_attempts`). [ref: gpu4agents/queue/coordinator.py#L1309-L1311] [ref: gpu4agents/queue/coordinator.py#L1416-L1419] [ref: gpu4agents/queue/backends/local.py#L236-L239]
- Requeue after a machine fault waits for proof the run stopped (destroy succeeded, or deadline plus grace for an unseen machine) instead of a timeout guess; cost: a job on another host's machine waits up to its deadline plus 300 s. [ref: gpu4agents/queue/coordinator.py#L84-L84] [ref: gpu4agents/queue/coordinator.py#L1119-L1120] [ref: gpu4agents/queue/coordinator.py#L866-L879]
- Fencing (`refuse`) was added so a never-started run can go back immediately without risking a late start from the dead coordinator (alternative before: wait out the timing rules, still the fallback when the fence fails). [ref: commit:d14e8e0] [ref: gpu4agents/queue/coordinator.py#L814-L829]
- Lease and age times come from the backend's own `now()`: the SQLite backend's clock (the host's `time.time` by default) and the Redis server's `TIME`. [ref: gpu4agents/queue/backends/local.py#L44-L53] [ref: gpu4agents/queue/backends/local.py#L122-L122] [ref: gpu4agents/queue/backends/redis.py#L343-L345]
- Secrets never live in `gpu.yaml`: `queue.backend`/`queue.outputs` reject any value containing `://`. [ref: gpu4agents/core/config.py#L425-L434]

### Agent surfaces and CLI

- MCP tools shell out to the CLI with `--json` instead of calling `Engine` in-process; the module docstring gives the reason (same output, exit codes and file lock as an application). Cost: one Python subprocess per tool call and a 900 s ceiling per call. [ref: gpu4agents/agent/mcp.py#L5-L7] [ref: gpu4agents/agent/mcp.py#L204-L226]
- Setup runs detached from the MCP server so it outlives a client's tool timeout and the client itself; the agent re-attaches by calling gpu_setup again, and the job is keyed per project so a repeated call never starts a second setup. Test: [ref: tests/test_agent.py#L161-L183] Code: [ref: gpu4agents/agent/mcp.py#L288-L297]
- The destroy confirmation lives in the MCP layer (`confirm=true`), and the MCP server then passes `--yes` to the CLI, whose interactive prompt is skipped under `--json` anyway. [ref: gpu4agents/agent/mcp.py#L430-L440] [ref: gpu4agents/cli/lifecycle.py#L95-L103]
- The dashboard's mutating routes are off by default and, when on, go through the `gpu` CLI rather than a provider client, so the read-only mode needs no credentials and control mode does (the code comment in `_build_actions` says so). [ref: gpu4agents/core/config.py#L369-L369] [ref: gpu4agents/cli/dashboard.py#L72-L83] [ref: gpu4agents/cli/dashboard.py#L101-L107] [ref: gpu4agents/cli/dashboard.py#L139-L155]
- The package installs the same entry point under two names, `gpu` and `gpu4agents`; the pyproject comment ties this to Oh My Zsh's `gpu` alias. [ref: pyproject.toml#L54-L57]
- Codex's config.toml is edited as text (the `[mcp_servers.gpu]` table is replaced up to the next table header) rather than parsed; the docstring cites Python 3.10 having no TOML parser. [ref: gpu4agents/agent/install.py#L125-L157]

## 5. What failed and what we changed

Each failure below names the fix commit and, where one exists, the test that now pins the behaviour. Symptoms and causes come from the fix commit or from the document that recorded the failure.

### A vanished machine was billed until someone noticed, and blamed on the wrong rule
- Symptom, as the commit records it: a project left without a provider key reconciled, days later, a machine that the provider had stopped billing long before; the reconcile billed the whole gap and recorded `max_lifetime` as the reason, although the idle rule had come due first. [ref: commit:1892cf2]
- Cause, as the diff shows it: the reconcile closed the open billable period at the moment of the check, and judged the lifecycle rules as of that same moment, where `max_lifetime` takes priority. [ref: commit:1892cf2]
- Fix: every reconcile that finds the machine at the provider stamps `last_seen`. [ref: gpu4agents/core/engine.py#L416-L417]
- Fix: `_why_missing` replays the rules forward from the last moment the machine was known to exist and returns the first destroy that comes due, with its time. [ref: gpu4agents/core/engine.py#L320-L339]
- Fix: when a stop (not a destroy) comes due first, no rule is credited, since a stopped machine would still exist. [ref: gpu4agents/core/engine.py#L335-L336]
- Fix: the billable period closes at that rule's due time, or at the last known existence when no rule was due. [ref: gpu4agents/core/engine.py#L380-L383]
- Fix: the worst case (billed until now) is still computed and recorded in the event. [ref: gpu4agents/core/engine.py#L375-L379] [ref: gpu4agents/core/engine.py#L400-L400]
- Fix: when a rule is inferred, `instance_destroyed` records `reason` and `by` as `unknown` and keeps the inference in separate fields with `inferred=True`. [ref: gpu4agents/core/engine.py#L403-L413]
- Fix: when the provider cannot be asked, reconcile keeps the saved state but records the error, and `status` reports `reconciled: false` with a `stale_reason`. [ref: gpu4agents/core/engine.py#L362-L367] [ref: gpu4agents/core/engine.py#L2007-L2015]
- Fix: the human `gpu status` view prints a red `STALE` line in that case (when no other command holds the lock). [ref: gpu4agents/cli/status.py#L77-L86]
- Pinned by: the first-rule replay of the field timeline, billed until the idle rule's due time. [ref: tests/test_field_run.py#L481-L516]
- Pinned by: no rule due, billed until last seen. [ref: tests/test_field_run.py#L519-L536]
- Pinned by: reconcile stamps `last_seen`. [ref: tests/test_field_run.py#L539-L544]
- Pinned by: status says it is stale when the provider cannot be asked. [ref: tests/test_field_run.py#L547-L567]
- Pinned by: the inferred event's `unknown` reason and separate `likely_reason`. [ref: tests/test_field_run.py#L470-L478]

### vLLM crashed on GPUs too old for the image (Vast)
- Symptom and cause, as the commit records them: the offer search set a floor on the host driver's CUDA (`cuda_max_good`) but not on GPU architecture, so Pascal/Volta cards behind a CUDA-13 driver passed the filter and vLLM crashed at engine-core init. Both live serve attempts landed on a GTX 1080 and a V100. [ref: commit:f372274]
- Fix: the vLLM runtime sets an architecture floor from the image's CUDA: 8.0 when the image's CUDA is 13 or newer, 7.0 below that, and none when the CUDA cannot be read from the tag. [ref: gpu4agents/runtime/vllm.py#L134-L142]
- Fix: the engine uses the higher of `compute.gpu.compute_cap_min` and the runtime's floor. [ref: gpu4agents/core/engine.py#L557-L567]
- Fix: the Vast query sends the floor as `compute_cap` ×100, and `offer_matches` also rejects any offer whose reported capability is below it. [ref: gpu4agents/providers/vast/mapper.py#L221-L222] [ref: gpu4agents/providers/vast/mapper.py#L288-L291]
- Pinned by: a V100 (700) and a GTX 1080 (610) are rejected under an 8.0 floor, and an offer with no reported capability passes. [ref: tests/test_vast_mapper.py#L182-L191] The runtime floors per image are pinned in [ref: tests/test_runtime_vllm.py#L244-L254].

### A model server that died at boot made setup wait out the whole startup timeout
- Symptom and cause, as the commit records them: `vllm/vllm-openai:latest` (CUDA 13.0) landed on a 12.2 driver, vLLM died at boot, and setup waited out the whole startup timeout with no way to see why. [ref: commit:4f0a79a]
- Fix: while it waits for health, the engine asks the watchdog whether the runtime is alive. When `runtime_alive` is `False`, it returns a reason that includes the last lines of the runtime log, and that reason aborts the wait. [ref: gpu4agents/core/engine.py#L1235-L1251] [ref: gpu4agents/core/engine.py#L1263-L1280]
- Fix (watchdog side): a dead runtime during startup ends the startup grace and logs `runtime_exited`. [ref: gpu4agents/watchdog/agent_body.py#L251-L257]
- Pinned by: setup raises `RuntimeStartupError` in under 10 s, quotes the log, asks for 30 lines of the runtime log and destroys the instance. [ref: tests/test_engine.py#L557-L574]

### A bad marketplace host lost the whole setup
- Symptom, as the commits record it: setups were lost to a host that could not reach Docker Hub (Vast status FAILED) and to one that stayed in `loading` past the startup timeout. [ref: commit:83ab1de] A host that failed an image pull was picked again by the next setup, then killed vLLM mid-download, and setup reported that as a model failure and gave up. [ref: commit:25a7a7f]
- Fix: on `HostFailure`, setup destroys the instance, excludes its host and tries again up to `compute.host_attempts` times. Retries need `cleanup_on_setup_failure`. A host it could not destroy stops the retries. The last failed host is also marked bad. [ref: gpu4agents/core/engine.py#L825-L825] [ref: gpu4agents/core/engine.py#L854-L873] [ref: gpu4agents/core/engine.py#L979-L1001]
- Fix: a dead model server whose log matches the host-network failure pattern raises `HostFailure`, so another host is tried, not a model failure. [ref: gpu4agents/core/engine.py#L1235-L1243]
- Pinned by: a host that fails to boot is replaced, and only one instance is alive at a time. [ref: tests/test_engine.py#L644-L665] Failed hosts are remembered by the next setup and the record expires after its TTL. [ref: tests/test_engine.py#L894-L919]

### Stalled image pulls held a host until the boot timeout
- Symptom, as the commit records it: live, three hosts froze on one pull line, or kept retrying one layer, until the 30-minute `boot_timeout`, and none recovered. [ref: commit:97b680c]
- Fix: when the provider reports boot progress, pull progress in the boot message that does not move for `pull_stall_timeout` raises a `HostFailure` whose reason is "image pull stalled for ...". [ref: gpu4agents/core/engine.py#L1096-L1099] [ref: gpu4agents/core/engine.py#L1135-L1146]
- Pinned by: a frozen pull is replaced after about 12 simulated minutes, not 30, and the stalled host is not rented again. [ref: tests/test_engine.py#L736-L752]

### Vast rate limits failed setup
- Symptom, as the commit records it: a setup that ran while the sweep timer was running failed on the offer search. [ref: commit:3237f6e]
- Fix: the Vast client resends a 429 for any method, and a 502/503/504 only for GET, sleeping for `Retry-After` (capped at 60 s) or the configured delay. [ref: gpu4agents/providers/vast/client.py#L86-L101] [ref: gpu4agents/providers/vast/client.py#L35-L37]
- Pinned by: two 429s then a 200 succeed after sleeping 3.0 s and 4.0 s, and a 429 that never lifts becomes a `ProviderError`. [ref: tests/test_vast_provider.py#L383-L408]

### The boot script went over Vast's 16 KB onstart limit
- Symptom, as the commit records it: the hardened watchdog pushed the onstart script to 18 KB and Vast refused the create. [ref: commit:d953aea]
- Fix: the watchdog source is parsed and unparsed with its docstrings removed (`ast.unparse`, which also drops comments) before it is embedded. [ref: gpu4agents/watchdog/builder.py#L60-L88] The Vast provider raises `ProvisioningError` before any API call when the script is over `ONSTART_MAX_BYTES`. [ref: gpu4agents/providers/vast/provider.py#L286-L292]
- Pinned by: the real boot script must be more than 400 bytes under the limit, and an oversized script makes zero HTTP calls. [ref: tests/test_vast_provider.py#L301-L326]

### A Vast instance that was stopping was treated as READY
- Symptom and cause, as the commit records them: seconds after a stop, Vast still reports `actual_status=running` while `cur_state`/`intended_status` say stopped. Mapping that to READY made a setup right after a stop skip the restart. [ref: commit:28e2030]
- Fix: unless `status_msg` reports an error, `map_state` returns STOPPED when `intended_status` or `cur_state` is `stopped`, before it looks at `actual_status`. [ref: gpu4agents/providers/vast/mapper.py#L34-L48]
- Pinned by: [ref: tests/test_vast_mapper.py#L268-L272]

### A short provider API outage made the watchdog destroy the disk
- Symptom and cause, as the commit records them: three failed stops (three minutes of a Vast API outage) escalated to destroy, and `stop_failures` never reset, so failures from separate idle periods added up. [ref: commit:77ea1e4]
- Fix: failures reset when the decision is no longer a shutdown. A stop escalates to destroy only after `escalate_after_s` of continuous failure (default 1800 s), and only when destroy is supported. [ref: gpu4agents/watchdog/agent_body.py#L410-L427] [ref: gpu4agents/watchdog/agent_body.py#L88-L88]
- Pinned by: five failed stops do not destroy, failures reset when the machine is back in use, 1801 s of failure escalates, and nothing escalates without destroy support. [ref: tests/test_watchdog.py#L613-L656]

### Job results could be deleted before anyone downloaded them
- Recorded in: CHANGELOG 0.12.0 "Fixed" says that once 0.11.0 made an idle Vast machine be destroyed, a detached job's outputs that nobody had pulled were destroyed with it. [ref: CHANGELOG.md#L59-L65] Fix commit: [ref: commit:f9c9a9f]
- Fix: a watchdog terminate hook (`outputs_guard`) turns a destroy into a stop when results are unsaved and the stop keeps the disk. With no lasting disk, it holds an idle shutdown for `OUTPUTS_HOLD_S` after the job ends. [ref: gpu4agents/watchdog/jobs_ext.py#L201-L222] It is registered in [ref: gpu4agents/watchdog/jobs_ext.py#L691-L691] and called in [ref: gpu4agents/watchdog/agent_body.py#L463-L470].
- Fix (CLI): `destroy` downloads job outputs first. When it cannot, it raises `StateError` unless `discard_outputs` is set. [ref: gpu4agents/core/engine.py#L2238-L2247]
- Pinned by: [ref: tests/test_job_outputs.py#L36-L43] [ref: tests/test_job_outputs.py#L187-L201]

### The agent's own `gpu stop` left a machine billing disk, with nothing to destroy it (field use, 0.12.0)
- Recorded in: handoff item G1. The agent ran `gpu stop`, and the stopped Vast machine kept billing disk with no sweep timer installed to enforce `stopped_ttl`. [ref: docs/handoff/2026-11-11-video-validation-fixes.md#L30-L39] Fix commit: [ref: commit:8af328a]
- Fix: on a provider where stopping does not end billing, `stop()` adds `still_billing`, the daily disk cost, whether the sweep timer is installed, and a warning that names `gpu destroy`. [ref: gpu4agents/core/engine.py#L2099-L2101] [ref: gpu4agents/core/engine.py#L2202-L2230]
- Fix: CLI commands (except mcp, ls, sweep, gateway, dashboard and queue) warn about stopped machines that are still billing, read from state files; with the sweep timer on, only machines past `stopped_ttl` are named. `GPU_NO_STOPPED_CHECK` turns it off. [ref: gpu4agents/cli/_common.py#L144-L174] [ref: gpu4agents/core/sweep.py#L586-L616] [ref: gpu4agents/cli/main.py#L23-L23] The call site is [ref: gpu4agents/cli/main.py#L69-L69].
- Pinned by: [ref: tests/test_field_run.py#L71-L89] An acceptance test runs a manual stop, then only the timer's sweep, and the sweep destroys the machine for `stopped_ttl`. [ref: tests/test_field_run.py#L130-L158] A command run in another directory prints "still billing". [ref: tests/test_field_run.py#L161-L178]

### The project was kept in a session scratchpad, which left an orphan (field use)
- Recorded in: handoff item G2. Both filmed takes created `gpu.yaml` under the agent session's scratchpad, and a later session saw the machine as an orphan it could not use. [ref: docs/handoff/2026-11-11-video-validation-fixes.md#L53-L57] Fix commit: [ref: commit:8af328a]
- Fix: `ephemeral_reason` flags any path with a `scratchpad` component, or under `/tmp`, `/var/tmp`, `/dev/shm` or the system temp dir, unless `GPU_ALLOW_TEMP_PROJECT` is set. [ref: gpu4agents/core/paths.py#L45-L72] `gpu init` exits 2 on such a directory unless `--allow-temp` is passed. [ref: gpu4agents/cli/setup.py#L91-L105]
- Pinned by: [ref: tests/test_field_run.py#L231-L254]

### The card the user approved was not the card that was rented (field use)
- Recorded in: handoff item G4. The approval quoted one card, and after a host retry a different, older card was rented. The agent noticed only in its final report. [ref: docs/handoff/2026-11-11-video-validation-fixes.md#L85-L89] Fix commit: [ref: commit:8af328a]
- Fix: a pinned setup (`offer_id`) sets the name filter to the quoted GPU and caps the price at the quote plus a rounding tolerance. If no matching offer is left, it raises `CapacityUnavailableError` without renting. [ref: gpu4agents/core/engine.py#L736-L745] [ref: gpu4agents/core/engine.py#L831-L840]
- Fix: `rented_vs_quoted` sets `matches_quote: false` and a warning when the rented card differs or costs more than the quote plus tolerance. [ref: gpu4agents/core/engine.py#L747-L770]
- Pinned by: a pinned setup never falls back to another card or a higher price, a pinned host retry stays on the quoted card, and an unpinned setup reports the mismatch. [ref: tests/test_field_run.py#L375-L429]

### A coordinator crash before dispatch cost an upload timeout and an attempt (queue field run)
- Recorded in: the queue field run. The crash landed between recording an attempt and sending it. The next coordinator waited out the upload timeout, the idle rule ended the assigned machine meanwhile, and the job used up an attempt. [ref: docs/handoff/2026-11-12-queue-mode-field-run.md#L24-L29] Fix commits: [ref: commit:0a3d9e1] [ref: commit:d14e8e0] [ref: commit:e6523a2]
- Fix: when a coordinator takes over and the home machine shows the run never started, it fences the run first, then requeues it as `NOT_STARTED`. If the fence cannot be confirmed, the timing rules decide, and a run still in doubt keeps its machine awake. [ref: gpu4agents/queue/coordinator.py#L801-L829] [ref: gpu4agents/queue/coordinator.py#L831-L847]
- Fix (machine side): `POST /jobs/<id>/refuse` records the id as `refused` and removes its bundle and source. A job that already started is returned as it is. [ref: gpu4agents/watchdog/jobs_ext.py#L280-L295]
- Fix: `NOT_STARTED` attempts do not count toward `max_attempts`, but more than `NOT_STARTED_FACTOR` (2) × `max_attempts` of them fail the job. [ref: gpu4agents/queue/coordinator.py#L1414-L1419] [ref: gpu4agents/queue/coordinator.py#L89-L89]
- Fix: a restarted watchdog session clears `deadline_at`, so a pool machine started again is not ended by its old deadline. [ref: gpu4agents/watchdog/agent_body.py#L190-L202]
- Pinned by: [ref: tests/test_queue_coordinator.py#L792-L807] [ref: tests/test_queue_coordinator.py#L846-L891] [ref: tests/test_queue_coordinator.py#L933-L954] [ref: tests/test_watchdog.py#L488-L502]

### The dashboard marked a live GPU as a ghost
- Symptom and cause, as the commit diff shows them: `live_ids` read `i["id"]`, but `gpu --json ls` rows carry `instance_id`, so live machines were not matched. [ref: commit:4721425] The `ls` rows key the id as `instance_id` [ref: gpu4agents/core/sweep.py#L649-L651], and `ls --json` prints those rows as they are. [ref: gpu4agents/cli/account.py#L111-L115]
- Fix: `live_ids` reads `instance_id`, falls back to `id`, and returns None (unverified) on a non-zero exit or any exception. [ref: gpu4agents/dashboard/ops.py#L34-L50]
- Pinned by: [ref: tests/test_dashboard.py#L137-L151]

### The dashboard's control buttons put the project name inside a JS string
- Cause, as the commit diff shows it: the button markup built `onclick="doAction('<action>','<project>')"`, and `esc()` did not escape `'`. [ref: commit:8c0478b]
- Fix: `esc()` now escapes `'`. Buttons carry the action and project in HTML-escaped `data-*` attributes, and one delegated click listener reads them. [ref: gpu4agents/dashboard/page.html#L94-L95] [ref: gpu4agents/dashboard/page.html#L126-L134] [ref: gpu4agents/dashboard/page.html#L282-L283]
- Fix (server side, as it stands now): a control POST for a project not in `project_dirs` gets 409. [ref: gpu4agents/dashboard/server.py#L163-L171] Pinned by [ref: tests/test_dashboard.py#L403-L425].

### The llama engine's first live boot failed
- Recorded in: the design spec's live validation. The first boot surfaced two bugs: the launcher called `llama-server`, which is not on PATH (`/app/llama-server`), and the pinned image digest rejected the `qwen35moe` architecture at load. [ref: docs/superpowers/specs/2026-11-15-llama-engine-anthropic-adapter-design.md#L186-L191]
- Fix (binary): the launcher resolves `command -v llama-server` and falls back to `/app/llama-server`. [ref: commit:7114d98] [ref: gpu4agents/runtime/llama.py#L123-L129] Pinned by [ref: tests/test_llama_runtime.py#L145-L150].
- Fix (image): `DEFAULT_IMAGE` is pinned to a newer `server-cuda` digest. [ref: commit:2c46113] [ref: gpu4agents/runtime/llama.py#L41-L44] No test in tests/ references the image or digest, so the bump is not pinned by a test.
- Commit 7114d98 also sets `LLAMA_CACHE` under `/workspace` [ref: commit:7114d98] [ref: gpu4agents/runtime/llama.py#L74-L81], pinned by [ref: tests/test_llama_runtime.py#L153-L156]. Its test section header calls these "review fixes". [ref: tests/test_llama_runtime.py#L143-L143]

### Gaps in the Anthropic-to-OpenAI gateway adapter
- Query string (field): Claude Code calls `/v1/messages?beta=true`. The old `target_path` checked `path.rstrip("/").endswith("/v1/messages")`, which a query string defeats, so the path was not rewritten. [ref: commit:3743af1] The fix splits off the query, rewrites the base and puts the query back. [ref: gpu4agents/gateway/anthropic_openai.py#L266-L274] Pinned by [ref: tests/test_anthropic_openai.py#L355-L367].
- Review fixes in [ref: commit:7114d98] (test header: [ref: tests/test_anthropic_openai.py#L306-L306]):
  - An assistant turn with no text and no tool calls now sends `content: ""`, not null. [ref: gpu4agents/gateway/anthropic_openai.py#L104-L111]
  - A 200 response whose body is `{"error": ...}` with no `choices` becomes an Anthropic error envelope. [ref: gpu4agents/gateway/anthropic_openai.py#L299-L302]
  - The streamed `message_delta` carries `input_tokens`. [ref: gpu4agents/gateway/anthropic_openai.py#L438-L438]
  - Pinned by [ref: tests/test_anthropic_openai.py#L308-L350].

### A restarted instance stayed STOPPED locally and was never guarded again
- Cause: reconcile took the watchdog's `terminated` flag over the provider's state. Fix: the provider decides running vs stopped, and `terminated` matters only when the provider reports UNKNOWN. On the watchdog side, a new session id resets `terminated`. [ref: commit:c989468] [ref: gpu4agents/core/engine.py#L459-L469] [ref: gpu4agents/watchdog/agent_body.py#L176-L203]
  - Pinned by: [ref: tests/test_engine_sessions.py#L37-L66], [ref: tests/test_engine_sessions.py#L69-L79], [ref: tests/test_watchdog.py#L462-L485].

### The sweep destroyed another project's locked, in-use instance
- Cause: `sweep --apply` judged every untracked instance by the current project's thresholds and by provider creation age. Fix: each registered project is guarded by its own `guard_tick`; only true orphans get the blunt rules, and a running orphan is only stopped. The same commit stopped the CLI from applying `idle_timeout` when an installed watchdog is unreachable (the pinning test's docstring records that it had stopped an endpoint that was serving requests). [ref: commit:48c5ff9] [ref: gpu4agents/core/sweep.py#L322-L371] [ref: gpu4agents/core/engine.py#L178-L192]
  - Pinned by: [ref: tests/test_sweep.py#L354-L373], [ref: tests/test_sweep.py#L376-L396], [ref: tests/test_sweep.py#L135-L146], [ref: tests/test_engine_sessions.py#L261-L277].

### A stopped instance was reported as `idle_skipped`
- Fix: `_idle_blind` also requires a billable-compute state. [ref: commit:1d81909] [ref: gpu4agents/core/engine.py#L190-L191] [ref: tests/test_engine_sessions.py#L312-L323]

### A fence during an upload broke the upload handler
- Symptom: a fence arriving during a short upload could break the upload handler, and a failed bundle move was reported as a 409 conflict (read as "already started"). Fix: tolerant removal of the `.part` file and a 500 for a failed move. [ref: commit:1beba37] [ref: gpu4agents/watchdog/jobs_ext.py#L256-L268] Tests: [ref: tests/test_jobs.py#L379-L399]

### Retiring a machine counted already-downloaded outputs as lost
- Symptom: retiring a machine counted a run whose outputs the coordinator had already downloaded as lost. Fix: after a fetch, a run the runner reports as ended and saved is removed from the machine's job list. [ref: commit:8a3f6bb] [ref: gpu4agents/queue/pool.py#L602-L616] Test: [ref: tests/test_queue_pool.py#L528-L540]

### A dropped connection during the weight download cost the whole host
- Symptom: a dropped connection during the weight download killed vLLM and cost the whole host. Fix: a launcher loop that restarts the server in place on network signatures. If the network keeps failing, the engine raises `HostFailure` and tries another host. [ref: commit:f15adab] Tests: [ref: tests/test_runtime_vllm.py#L312-L334] [ref: tests/test_engine.py#L882-L891]

### Ctrl-C during setup left the machine booting and billing
- Symptom: Ctrl-C during setup left the instance booting and billing. Fix: `KeyboardInterrupt` is routed to `_fail_setup`. [ref: commit:7278cfe] [ref: gpu4agents/core/engine.py#L874-L878] Test: [ref: tests/test_engine.py#L114-L125]

## 6. Limits and open problems

### Providers

- Only Vast honours `preferred_offer_id` when sorting offers [ref: gpu4agents/providers/vast/provider.py#L283-L283]. RunPod sorts by price, then stock, then id, and Lambda by price, then id [ref: gpu4agents/providers/runpod/provider.py#L266-L266] [ref: gpu4agents/providers/lambdacloud/provider.py#L249-L249]. On those two, a pinned offer narrows the search only through `_pin`'s name filter and price cap [ref: gpu4agents/core/engine.py#L741-L745].
- RunPod and Lambda offers never set `host_id` [ref: gpu4agents/providers/runpod/mapper.py#L77-L89] [ref: gpu4agents/providers/lambdacloud/mapper.py#L74-L85], and their `offer_matches` never reads `exclude_hosts` [ref: gpu4agents/providers/runpod/mapper.py#L99-L114] [ref: gpu4agents/providers/lambdacloud/mapper.py#L90-L111]. Bad-host exclusion therefore only takes effect on Vast, whose `offer_matches` drops offers from excluded hosts [ref: gpu4agents/providers/vast/mapper.py#L300-L301].
- RunPod rejects spot instances and remote workloads in `validate`, and `run` always raises `UnsupportedCapabilityError` [ref: gpu4agents/providers/runpod/provider.py#L205-L208] [ref: gpu4agents/providers/runpod/provider.py#L337-L338].
- `stopped_and_billing` reads `stop_eliminates_billing` from the registered class, not from a configured instance, when no disk price was recorded [ref: gpu4agents/core/sweep.py#L586-L586] [ref: gpu4agents/core/sweep.py#L604-L611]. RunPod's class says True even though a `volume_gb` pod bills while stopped [ref: gpu4agents/providers/runpod/provider.py#L141-L154], and RunPod instances never get `stopped_hourly_cost_usd` [ref: gpu4agents/providers/runpod/mapper.py#L149-L161]. A stopped RunPod pod with a volume is therefore skipped by that check.
- `VastOptions.from_mapping` raises a bare `ValueError` on unknown options [ref: gpu4agents/providers/vast/config.py#L55-L64], and `VastProvider.__init__` does not convert it [ref: gpu4agents/providers/vast/provider.py#L150-L153]. RunPod and Lambda convert the same error to `ConfigError` [ref: gpu4agents/providers/runpod/provider.py#L146-L149] [ref: gpu4agents/providers/lambdacloud/provider.py#L154-L157].
- Vast's `validate` previews offers with a spec that omits `compute_cap_min` [ref: gpu4agents/providers/vast/provider.py#L254-L269], while RunPod and Lambda include it [ref: gpu4agents/providers/runpod/provider.py#L249-L249] [ref: gpu4agents/providers/lambdacloud/provider.py#L240-L240].

### Setup and serving

- The fingerprint omits the llama fields `runtime.quant`, `n_gpu_layers` and `llama_args`, all of which feed `llama-server`'s argv. A llama machine whose `quant` changed passes the fingerprint check and is reused with the old command. The fingerprint test does not cover these fields. [ref: gpu4agents/core/config.py#L613-L626] [ref: gpu4agents/runtime/llama.py#L88-L112] [ref: gpu4agents/core/engine.py#L789-L799] [ref: tests/test_config.py#L307-L319]
- The reuse path returns `ensure_ready` before the try/except that calls `_fail_setup`. A reused or restarted instance whose server fails to start is therefore not destroyed by `cleanup_on_setup_failure`. [ref: gpu4agents/core/engine.py#L796-L799] [ref: gpu4agents/core/engine.py#L852-L878]
- The model card never selects `llama`, and `gpu init --engine` rejects `llama`, so the llama engine is set by hand in gpu.yaml. [ref: gpu4agents/serve/__init__.py#L43-L45] [ref: gpu4agents/cli/setup.py#L116-L118]
- The llama argv ignores `model.served_name`, `model.revision` and `model.trust_remote_code`. [ref: gpu4agents/runtime/llama.py#L88-L112]
- `gpu redeploy` works only for `hf`/`custom`. [ref: gpu4agents/core/engine.py#L1596-L1600]
- The hf chat backend reads only messages, temperature, max tokens, top_p and stop (no tools). Streaming is emulated as one chunk followed by `[DONE]`. [ref: gpu4agents/serve/hf/server.py#L304-L342] [ref: gpu4agents/serve/hf/server.py#L790-L803]

### Watchdog and cost caps

- Nothing on a stopped machine enforces `stopped_ttl`: the watchdog's snapshot always says `provider_state="ready"`, and only the CLI snapshot carries `stopped_since`. That leaves `gpu guard`/`gpu sweep` on the user's computer. [ref: gpu4agents/watchdog/agent_body.py#L267-L278] [ref: gpu4agents/core/engine.py#L163-L176]
- `max_hourly_cost_usd` is never re-checked at runtime. It is used only before provisioning and in offer search; `evaluate` has no hourly cap. [ref: gpu4agents/core/engine.py#L632-L662] [ref: gpu4agents/core/engine.py#L530-L530] [ref: gpu4agents/core/policy.py#L206-L278]
- The probe supports only `kind: prometheus`; any other kind disables it. [ref: gpu4agents/watchdog/agent_body.py#L641-L647]
- A provider with no stop and no destroy gets ACTION_NONE (`max_lifetime_no_action` / `idle_no_action`). A running orphan on a provider that cannot stop is reported, not acted on. [ref: gpu4agents/core/policy.py#L192-L203] [ref: gpu4agents/core/policy.py#L247-L249] [ref: gpu4agents/core/policy.py#L272-L276] [ref: gpu4agents/core/sweep.py#L511-L514]
- When enabled, the watchdog requires provider actions: setup raises `UnsupportedCapabilityError` if the provider supplies no `watchdog_spec`. [ref: gpu4agents/core/engine.py#L590-L596]
- The sweep requires the `list_instances` capability. [ref: gpu4agents/core/sweep.py#L439-L443]
- `lifecycle.ssh_counts_as_activity` is declared in the config model, but nothing in `gpu4agents/` reads it. [ref: gpu4agents/core/config.py#L243-L243]
- `_why_missing` never credits a stop: if the first rule due was a stop, the likely reason is "unknown". [ref: gpu4agents/core/engine.py#L333-L336] [ref: gpu4agents/core/engine.py#L371-L371] [ref: gpu4agents/core/engine.py#L384-L390]

### Jobs and queue

- On direct `gpu job run`, `max_cost_usd` and `min_vram_gb` in `gpujob.yaml` are accepted by the model but `job_submit` never reads them; only the queue preflight uses them. [ref: gpu4agents/core/job.py#L86-L87] [ref: gpu4agents/core/engine.py#L1439-L1469] [ref: gpu4agents/queue/preflight.py#L66-L67]
- Jobs need the watchdog: `lifecycle.watchdog.enabled: false` makes every job call raise `ConfigError`; a stopped machine must be restarted with `gpu setup` for status/logs/cancel (only `gpu job pull` wakes it). [ref: gpu4agents/core/engine.py#L1385-L1401] [ref: gpu4agents/core/engine.py#L1685-L1693]
- Upload bundle max is 2 GiB on both sides; job ids must match `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`; `GET /jobs/<id>` lists at most 200 output files. [ref: gpu4agents/core/job.py#L68-L68] [ref: gpu4agents/watchdog/jobs_ext.py#L85-L90] [ref: gpu4agents/watchdog/jobs_ext.py#L244-L245]
- The Redis backend targets a single Redis (or primary with replicas), not Redis Cluster (stated in the module; scripts address keys by prefix with `keys=[]`). [ref: gpu4agents/queue/backends/redis.py#L15-L16] [ref: gpu4agents/queue/backends/redis.py#L340-L341]
- Output stores are `local`, `s3` (needs boto3 and `GPU_OUTPUTS_URL`) or `module:factory`; anything else raises `ConfigError`. [ref: gpu4agents/queue/outputs.py#L405-L428] [ref: gpu4agents/queue/outputs.py#L279-L281]

### Agent surfaces and CLI

- `gpu gateway install` and `gpu dashboard install` raise `NotImplementedError` on anything but Linux (systemd only). [ref: gpu4agents/gateway/service.py#L90-L93] [ref: gpu4agents/dashboard/service.py#L90-L93]
- The gateway handler has no token check and binds whatever host `gateway.listen` names (default `127.0.0.1:8080`); unlike the dashboard there is no non-loopback refusal. [ref: gpu4agents/gateway/server.py#L55-L125] [ref: gpu4agents/gateway/server.py#L131-L139] [ref: gpu4agents/core/config.py#L338-L338] [ref: gpu4agents/dashboard/server.py#L28-L34]
- On a loopback bind with no token, the dashboard's control routes need only `confirm: true`: the gate returns true for a `None` token and `guard_bind` only checks non-loopback hosts. [ref: gpu4agents/dashboard/auth.py#L24-L26] [ref: gpu4agents/dashboard/server.py#L28-L34] [ref: gpu4agents/dashboard/server.py#L150-L162]
- `gpu init --engine` and gpu_init's `engine` enum accept auto, vllm, hf, custom and none, while the config schema also accepts `llama`. [ref: gpu4agents/cli/setup.py#L116-L118] [ref: gpu4agents/agent/mcp.py#L568-L572] [ref: gpu4agents/core/config.py#L150-L150]
- gpu_init passes only `--model`, `--engine`, `--task` and `--force`; it has no way to pass `--provider`, `--allow-temp`, `--jobs` or `--flat`. [ref: gpu4agents/agent/mcp.py#L243-L248]

### Across the failure history

- The llama.cpp image is pinned by digest, and the digest bump (F16) has no test of its own: the only image test checks that the image is pinned rather than `:latest`, not which digest. A stale digest shows up only on a live boot. [ref: gpu4agents/runtime/llama.py#L41-L44] [ref: tests/test_llama_runtime.py#L48-L52]
- The compute-capability floor trusts the server-side filter when an offer reports no capability. [ref: gpu4agents/providers/vast/mapper.py#L289-L291] [ref: tests/test_vast_mapper.py#L190-L190]
- A 502/503/504 on a non-GET Vast call is not retried, because a create may already have gone through. [ref: gpu4agents/providers/vast/client.py#L93-L98]
- Pull-stall detection applies only on providers that report boot progress. Elsewhere the whole `boot_timeout` applies. [ref: gpu4agents/core/engine.py#L1097-L1099]

### Where the docs and the code disagree

These are statements about the documentation: each cites the doc and the code it disagrees with.

- **What happens when an adapter returns no watchdog spec.** The guide says that returning `None` means users run `gpu guard` and "`validate` will warn" [ref: docs/ADDING_A_PROVIDER.md#L140-L141]. In the code, setup raises `UnsupportedCapabilityError` when the watchdog is enabled and the spec is `None` [ref: gpu4agents/core/engine.py#L590-L596]. The engine's `validate` only warns when the watchdog is disabled [ref: gpu4agents/core/engine.py#L220-L224], and `watchdog_spec` is called nowhere else in `core/engine.py` (grep).
- **How strict the name-branching test is.** The guide says the test "fails the build if `core/` learns a provider's name" [ref: docs/ADDING_A_PROVIDER.md#L169-L170]. The test only searches for three spellings of `== "vast"` [ref: tests/test_engine.py#L314-L316].
- **Where published ports come from.** The guide says `ComputeSpec.ports` lists the ports to publish [ref: docs/ADDING_A_PROVIDER.md#L145-L147]. The built-in adapters (Vast, RunPod, Lambda) ignore `spec.ports` and recompute ports from the config [ref: gpu4agents/providers/vast/provider.py#L186-L192] [ref: gpu4agents/providers/vast/provider.py#L301-L301] [ref: gpu4agents/providers/runpod/provider.py#L185-L191] [ref: gpu4agents/providers/runpod/provider.py#L278-L278] [ref: gpu4agents/providers/lambdacloud/provider.py#L184-L190] [ref: gpu4agents/providers/lambdacloud/provider.py#L262-L262]. The engine fills `spec.ports` from `runtime.ports()` plus the watchdog port [ref: gpu4agents/core/engine.py#L569-L573].
- CONFIG.md calls `health.interval` (60s) the probe interval. The code clamps the probe interval to at most 30 s. [ref: docs/CONFIG.md#L250-L250] [ref: gpu4agents/core/config.py#L552-L553] [ref: gpu4agents/core/engine.py#L1245-L1248] [ref: gpu4agents/core/health.py#L43-L43]
- CONFIG.md says `trust_remote_code` passes `--trust-remote-code` "to the runtime" and that `served_name` maps to `--served-model-name`. The llama argv sets neither. [ref: docs/CONFIG.md#L134-L135] [ref: gpu4agents/runtime/llama.py#L88-L112]
- CONFIG.md marks `dtype` "vLLM only". The hf engine also passes it as `JAMBU_DTYPE`, and the hf server reads it. [ref: docs/CONFIG.md#L147-L147] [ref: gpu4agents/runtime/service.py#L114-L114] [ref: gpu4agents/serve/hf/server.py#L93-L93]
- The fingerprint docstring says it MUST cover every field that feeds the server command line. The material it hashes leaves out the llama fields. [ref: gpu4agents/core/config.py#L586-L587] [ref: gpu4agents/core/config.py#L613-L626]
- README says, without qualification, that pointing `model.id` at another model and re-running `gpu setup` swaps the model on the same GPU. For the default `vllm` engine, a model change alters the fingerprint, and `setup` refuses without `--recreate`. [ref: README.md#L126-L126] [ref: gpu4agents/core/config.py#L607-L612] [ref: gpu4agents/core/config.py#L628-L634] [ref: gpu4agents/core/engine.py#L789-L795]
- `docs/CONFIG.md` describes `max_lifetime` as a "Hard ceiling on instance age". The policy measures it from the session start, so a restarted instance gets a new lease. (CONFIG.md itself says "per session" a few lines later.) [ref: docs/CONFIG.md#L209-L209] [ref: gpu4agents/core/policy.py#L242-L244] [ref: docs/CONFIG.md#L231-L231]
- The route list in the `agent_body.py` module docstring leaves out `/identify`, `/starting` and `/deadline`, which `dispatch` serves. [ref: gpu4agents/watchdog/agent_body.py#L9-L22] [ref: gpu4agents/watchdog/agent_body.py#L869-L904] [ref: gpu4agents/watchdog/agent_body.py#L976-L989]
- The `docs/CONFIG.md` row for `max_session_cost_usd` says the remote guard shuts down when the session cost passes the cap. The code disables the cap for pool slots, and the remote guard has no price to compute the cost with until `/identify` delivers it. [ref: docs/CONFIG.md#L263-L263] [ref: gpu4agents/core/engine.py#L144-L144] [ref: gpu4agents/core/engine.py#L157-L157] [ref: gpu4agents/core/engine.py#L610-L610] [ref: gpu4agents/watchdog/agent_body.py#L876-L877] [ref: gpu4agents/core/policy.py#L253-L253]
- `docs/CONFIG.md` lists `ssh_counts_as_activity` as a setting, but no code reads it. [ref: docs/CONFIG.md#L219-L219] [ref: gpu4agents/core/config.py#L243-L243]
- `docs/LIFECYCLE.md` says the watchdog evaluates the policy "every 60s". That is only the default of the configurable `interval_s`. [ref: docs/LIFECYCLE.md#L12-L12] [ref: gpu4agents/watchdog/agent_body.py#L489-L489]
- QUEUE.md says a machine that "fails the health check right after a job" sends the job back to the queue; in code that requeue happens only when the job also exited non-zero (and not by timeout); after a clean exit the machine is retired but the job is delivered as succeeded. [ref: QUEUE.md#L137-L137] [ref: gpu4agents/queue/coordinator.py#L1258-L1258] [ref: gpu4agents/queue/coordinator.py#L1270-L1281] [ref: gpu4agents/queue/coordinator.py#L1303-L1305]
- JOBS.md's `gpujob.yaml` reference lists six keys (run, setup, env, timeout, exclude, label), as does `load_manifest`'s error text; the model also accepts `max_cost_usd` and `min_vram_gb`, which only the queue uses. [ref: JOBS.md#L31-L37] [ref: gpu4agents/core/job.py#L123-L123] [ref: gpu4agents/core/job.py#L86-L87]
- `_deliver`'s docstring says retries go "up to MAX_DELIVERY_FAILURES"; no such constant exists: machine-side failures use `MAX_FETCH_FAILURES` (5) and store failures retry with backoff without a cap. [ref: gpu4agents/queue/coordinator.py#L1241-L1242] [ref: gpu4agents/queue/coordinator.py#L90-L91] [ref: gpu4agents/queue/coordinator.py#L1319-L1323]
- The skill's frontmatter description says "Works with Vast.ai out of the box", while its body says Vast.ai, RunPod and Lambda are built in, and the registry registers all three. [ref: skills/gpu4agents/SKILL.md#L3-L3] [ref: skills/gpu4agents/SKILL.md#L10-L10] [ref: gpu4agents/providers/registry.py#L140-L149]
- `gpu providers` help text says "Vast.ai is built in" against the same three-provider registry. [ref: gpu4agents/cli/providers.py#L26-L30] [ref: gpu4agents/providers/registry.py#L146-L147]
- Both service modules' docstrings say "systemd on Linux, launchd on macOS", but `install_service` raises `NotImplementedError` off Linux; the dashboard's copy also says it runs `gpu gateway start` while its unit runs `dashboard start`. [ref: gpu4agents/gateway/service.py#L1-L2] [ref: gpu4agents/dashboard/service.py#L1-L2] [ref: gpu4agents/dashboard/service.py#L66-L66] [ref: gpu4agents/dashboard/service.py#L90-L93]
- The dashboard server docstring says it "never mutates anything", but `do_POST` runs control actions when `control_enabled`. [ref: gpu4agents/dashboard/server.py#L3-L7] [ref: gpu4agents/dashboard/server.py#L144-L179]
- `gpu dashboard`'s module docstring says it "never calls a provider API and holds no provider credential", but with `control_enabled` the state provider calls `ops.live_ids`, which runs `gpu --json ls`. [ref: gpu4agents/cli/dashboard.py#L9-L12] [ref: gpu4agents/cli/dashboard.py#L139-L140] [ref: gpu4agents/cli/dashboard.py#L77-L80] [ref: gpu4agents/dashboard/ops.py#L28-L31] [ref: gpu4agents/dashboard/ops.py#L34-L38]
- The control-mode startup message says "token + confirm required", but on loopback with no token configured only confirm is enforced (see Limits). [ref: gpu4agents/cli/dashboard.py#L155-L155] [ref: gpu4agents/dashboard/auth.py#L24-L26]
- The `GatewayConfig.plugin` comment says a file plugin path is relative to the config file; `_load_plugin` resolves it with `Path(target).resolve()`, i.e. relative to the process cwd. [ref: gpu4agents/core/config.py#L342-L344] [ref: gpu4agents/gateway/base.py#L113-L121]
- The skill says exit codes are reported the same way for the CLI and MCP tools (`exit_code` in the error), but gpu_run returns the command's non-zero exit code inside a normal, non-error result. [ref: skills/gpu4agents/SKILL.md#L160-L160] [ref: gpu4agents/agent/mcp.py#L376-L384] [ref: gpu4agents/agent/mcp.py#L890-L898]
- Test comment vs code: `test_control_rejects_foreign_project` says "The dashboard controls only its own project" [ref: tests/test_dashboard.py#L404-L405]. The server now refuses only projects missing from `project_dirs`, so any project this host manages can be controlled [ref: gpu4agents/dashboard/server.py#L164-L171]. The CHANGELOG matches the code ("stops / destroys / locks any project this host manages") [ref: CHANGELOG.md#L9-L9].
- Comment vs code (minor): the page comment says buttons appear "only for a project this host can resolve", and the check is `CONTROLLABLE.indexOf(row.project)`. The original fix in 8c0478b compared against a single `LOCAL` project. [ref: gpu4agents/dashboard/page.html#L126-L129] [ref: commit:8c0478b]

## 7. How to read the code

Start at `gpu4agents/cli/main.py` for the commands, then `gpu4agents/core/engine.py` (the `Engine` class: setup, reconcile, status, jobs, stop, destroy), `gpu4agents/core/policy.py` (the lifecycle rule), and `gpu4agents/watchdog/` (what runs on the machine). Providers live in `gpu4agents/providers/<name>/` (client, mapper, config, provider). Engines live in `gpu4agents/runtime/` and the servers gpu ships in `gpu4agents/serve/`. The queue is `gpu4agents/queue/`, the agent surfaces `gpu4agents/agent/` and `skills/gpu4agents/`, and the local services `gpu4agents/gateway/` and `gpu4agents/dashboard/`.

### Entry points: Providers

- `gpu4agents/providers/base.py` `GPUProvider`: the adapter contract [ref: gpu4agents/providers/base.py#L49-L72].
- `gpu4agents/providers/registry.py` `load_builtin_providers`: registers built-in and entry-point adapters [ref: gpu4agents/providers/registry.py#L140-L149].
- `gpu4agents/providers/registry.py` `ProviderRegistry.get`: builds the configured adapter with a credential resolver [ref: gpu4agents/providers/registry.py#L70-L83].
- `gpu4agents/providers/scaffold.py` `write_template`: generates a third-party adapter skeleton [ref: gpu4agents/providers/scaffold.py#L241-L262].
- `gpu4agents/core/credentials.py` `CredentialResolver.get`: looks up credentials by name [ref: gpu4agents/core/credentials.py#L55-L67].
- `gpu4agents/watchdog/builder.py` `build_agent_source`: injects the adapter's watchdog actions into the watchdog script [ref: gpu4agents/watchdog/builder.py#L82-L99].

### Entry points: Setup and serving

- `Engine.setup`: rent (or reuse/recreate) a machine and bring the runtime up. [ref: gpu4agents/core/engine.py#L772-L879]
- `Engine.ensure_ready`: start a stopped machine, wait for boot, connect the watchdog, deploy the service, wait for health. [ref: gpu4agents/core/engine.py#L1022-L1074]
- `Engine.connection`: the client contract (base_url, key, model, routes). [ref: gpu4agents/core/engine.py#L1295-L1309]
- `RuntimeConfig.fingerprint`: the machine's identity for reuse. [ref: gpu4agents/core/config.py#L579-L636]
- `serve.serving_for`: engine and task from an HF card. [ref: gpu4agents/serve/__init__.py#L30-L45]
- `serve/hf/server.py:main`: gpu's own pipeline server. [ref: gpu4agents/serve/hf/server.py#L951-L977]

### Entry points: Watchdog and cost caps

- `policy.evaluate`: the pure lifecycle decision. [ref: gpu4agents/core/policy.py#L206-L206]
- `agent_body.main`: loads the config and state, loads saved extensions, starts the HTTP server, the guard thread and the probe thread. [ref: gpu4agents/watchdog/agent_body.py#L1028-L1064]
- `builder.build_agent_source` / `build_agent_config` / `build_onstart_script`: assemble the remote script and the boot script. [ref: gpu4agents/watchdog/builder.py#L82-L82] [ref: gpu4agents/watchdog/builder.py#L102-L102] [ref: gpu4agents/watchdog/builder.py#L169-L169]
- `Engine.guard_tick`: the CLI-side guard behind `gpu guard`. [ref: gpu4agents/core/engine.py#L2296-L2296] [ref: gpu4agents/cli/lifecycle.py#L125-L125]
- `Engine.reconcile`: merges provider truth, watchdog state and billing. [ref: gpu4agents/core/engine.py#L341-L350]
- `sweep.guard_projects` and `sweep.sweep`: the account-wide sweep, called from `gpu sweep`. [ref: gpu4agents/cli/account.py#L286-L311]
- `timer.install_timer`: writes the systemd units or the LaunchAgent. [ref: gpu4agents/cli/timer.py#L176-L212]
- `WatchdogClient`: the CLI's HTTP client for the watchdog. [ref: gpu4agents/core/beacon.py#L23-L27]

### Entry points: Jobs and queue

- `gpu4agents/cli/job.py` `run_command`: `gpu job run`, submit, follow the log, pull outputs, exit with the job's code. [ref: gpu4agents/cli/job.py#L73-L173]
- `gpu4agents/core/engine.py` `Engine.job_submit` / `_submit`: pack, upload and start a job on the project's machine. [ref: gpu4agents/core/engine.py#L1423-L1539]
- `gpu4agents/watchdog/jobs_ext.py` `jobs_route`: the `/jobs` HTTP routes on the watchdog. [ref: gpu4agents/watchdog/jobs_ext.py#L636-L671]
- `gpu4agents/queue/api.py` `submit_job`, `wait`, `outputs`, `drain`, `audit_queue`: the Python queue API behind CLI and MCP. [ref: gpu4agents/queue/api.py#L129-L179] [ref: gpu4agents/queue/api.py#L238-L249] [ref: gpu4agents/queue/api.py#L306-L306] [ref: gpu4agents/queue/api.py#L437-L482]
- `gpu4agents/cli/queue.py` `run_command`: `gpu queue run`, the coordinator loop (or `--once`), stops on SIGINT/SIGTERM. [ref: gpu4agents/cli/queue.py#L325-L379]

### Entry points: Agent surfaces and CLI

- `gpu` and `gpu4agents` console scripts both call `gpu4agents.cli.main:main`; optional extras are `dev`, `queue-redis` and `queue-s3`. [ref: pyproject.toml#L41-L57]
- `python -m gpu4agents` calls the same `main`, which is how the MCP server invokes the CLI. [ref: gpu4agents/__main__.py#L1-L4] [ref: gpu4agents/agent/mcp.py#L204-L205]
- `gpu mcp` (`mcp_command`): runs `Server(project=...).serve()` on stdio. [ref: gpu4agents/cli/agent.py#L20-L31] [ref: gpu4agents/cli/main.py#L99-L100]
- `gpu claude` (`claude_command`): launches Claude Code against the project's gateway. [ref: gpu4agents/cli/claude.py#L88-L115] [ref: gpu4agents/cli/main.py#L98-L98]
- `gpu gateway start|install|uninstall|status|logs` and `gpu dashboard start|install|uninstall|status|logs`: foreground servers and their systemd user units. [ref: gpu4agents/cli/gateway.py#L35-L135] [ref: gpu4agents/cli/dashboard.py#L133-L243]

### Tests

The suite has 40 `test_*.py` files plus `conftest.py`, `fakes.py` and `queue_fakes.py` (a directory count, not a line range).

- The core is tested against an in-memory `FakeProvider(GPUProvider)` named `fake`. It advertises every capability except `persistent_disk`, including `stop_eliminates_billing=True` and `boot_progress=True`. [ref: tests/fakes.py#L35-L52] [ref: gpu4agents/core/models.py#L69-L86]
- `FakeProvider` simulates a marketplace: one offer per host (`m1`..`m5`), with per-host prices and GPUs, and hosts scripted to fail to boot in three ways (`failed`, `vanish`, `loading`) with a scripted pull message per poll. [ref: tests/fakes.py#L62-L78]
- Field-run tests subclass the fake to model a provider whose stopped machine still bills (`BillingStopProvider`). [ref: tests/test_field_run.py#L34-L44]
- An autouse fixture points `GPU_HOME` at a temp dir and sets `GPU_NO_MODEL_CARD=1` and `GPU_ALLOW_TEMP_PROJECT=1`, because pytest's `tmp_path` is under /tmp, where `gpu init` would refuse. [ref: tests/conftest.py#L30-L40] The temp-folder tests unset it explicitly. [ref: tests/test_field_run.py#L234-L234] [ref: tests/test_field_run.py#L242-L242]
- `write_config` writes a namespaced `gpu.yaml` (`gpu_runtime:` wrapper) from a base config merged with per-test overrides, then loads it. [ref: tests/conftest.py#L53-L67]
- Watchdog tests run the real generated watchdog as a subprocess. They write `build_agent_source(actions)` to `agent.py` and start it with `sys.executable`. [ref: tests/test_watchdog.py#L121-L121] [ref: tests/test_watchdog.py#L140-L152]
- The queue coordinator and auditor tests use `FakePool`, an in-memory Pool with scripted job outcomes and machines that can vanish, hang or fail health checks. [ref: tests/queue_fakes.py#L1-L10] [ref: tests/queue_fakes.py#L77-L88] [ref: tests/queue_fakes.py#L147-L155] [ref: tests/queue_fakes.py#L228-L235]
- HTTP providers are faked at the transport level with `respx`, e.g. Vast. [ref: tests/test_vast_provider.py#L9-L9] Six test files reference respx: test_cli, test_lambda_provider, test_model_inspect, test_runpod_provider, test_sweep and test_vast_provider (from `grep -l respx tests/`).
