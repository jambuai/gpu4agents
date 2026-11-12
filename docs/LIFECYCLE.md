# How shutdown works

The CLI process cannot be responsible for idle shutdown. Close the laptop and a local watchdog dies while the GPU keeps billing. `setup` installs a watchdog on the instance itself:

```
CLI ──provisions──> instance
                      │
                      ├── supervisor (bash respawn loop)
                      │     └── watchdog (HTTP :8777, authenticated)
                      │           ├── holds the activity and heartbeat state
                      │           ├── reads vLLM /metrics for inference traffic
                      │           ├── evaluates the lifecycle policy every 60s
                      │           └── calls the provider API to stop or destroy itself
                      └── vLLM :8000
```

The watchdog script is generated at provision time. The lifecycle policy source is inlined from `gpu4agents/core/policy.py`, so the remote guard and `gpu status` apply the same rule. The provider adapter injects only the two calls that stop or destroy the machine.

An instance stays billable only while:

1. a workload is running (heartbeat alive), or
2. a workload is queued, or the model is still starting, or
3. the model server is serving requests, or
4. an explicit lock is active.

Anything else becomes `STOPPED` or `DESTROYED`, with the reason recorded.

Activity is a running workload, a queued workload, model startup, inference traffic, or an active lock. A healthy vLLM server, a health check, a metrics scrape, and an open SSH session are not activity. If they were, an idle server would stay up forever.

A crashed CLI, a dropped SSH session, or a sleeping laptop stops the heartbeat. After `lifecycle.heartbeat_grace` the workload is stale, the instance goes idle, and the normal idle rules apply.

## Inference traffic is activity

An agent that calls the vLLM endpoint directly, without `gpu run`, is still using the machine. The watchdog reads vLLM's Prometheus endpoint (`http://127.0.0.1:8000/metrics`) every `min(interval, 15s)` and marks activity when:

- a gauge is above zero: `vllm:num_requests_running`, `vllm:num_requests_waiting`, or
- a counter grew since the last successful read: `vllm:request_success_total`, `vllm:prompt_tokens_total`.

The first read only sets the baseline, so totals left over from earlier traffic do not keep the machine up. A counter that goes down means vLLM restarted; it rebaselines and does not count. A probe error is never activity, and it is logged as `activity_probe_error` at most once every 10 minutes. `GET /state` shows the last read under `last_probe` (`ok`, `values`, `last_traffic_at`).

Traffic only feeds the idle timer. It does not override `max_lifetime`, the session budget, or locks.

## Sessions and restarts

On the built-in provider (Vast.ai), a stop keeps the container disk, and the onstart script runs again on every container start. The watchdog state in `/var/lib/jambu/activity.json` therefore survives a stop/start, including the flag that says the instance was already stopped.

The onstart script sets `JAMBU_SESSION_STARTED_AT=$(date +%s)` once per container start and passes it to the watchdog. On boot the watchdog compares it with the stored `session_started_at`:

- Different (the instance was restarted): new session. The guard is re-armed, startup grace starts over, `last_activity` is now, workloads left running or queued are marked failed, and retry counters reset. Logged as `session_started`.
- Equal (the supervisor respawned a crashed watchdog in the same container start): state is kept.

`max_lifetime`, the session budget (`hourly_cost × session_age`) and startup grace are measured from the session start, not from creation. A restarted instance gets a fresh lease and is not killed the moment it boots. `GET /state` reports both `age_seconds` (since creation) and `session_age_seconds`.

## Retries and escalation

A failed provider call never disarms the guard. A transient API error is logged as `provider_error`, then `guard_retry` with the attempt count, and the guard tries again on its next tick.

When stops have failed continuously for `escalate_after_s` (watchdog config, default 30 minutes), the watchdog destroys the instance instead, if the provider supports destroy (reason `stop_failed_escalation`). A machine that bills forever costs more than a lost disk, but a few minutes of provider API outage is not worth the disk. The failure count and clock reset after a successful call and whenever the decision stops being a shutdown (the instance is in use again), so failures from separate idle spells never add up. A completed manual shutdown clears its pending request.

A successful call does not end the loop either. If the provider accepted the call but the machine is still running after `max(300s, 5 × interval)`, the watchdog issues it again (`guard_reissue`). A failed manual `POST /shutdown` is retried the same way.

## The supervisor

The watchdog runs inside a bash respawn loop started with `nohup`. If the watchdog crashes, the loop logs its exit code to `/var/log/jambu/watchdog.log` and starts it again 5 seconds later. `/var/run/jambu/watchdog.pid` holds the loop's pid; sending it `TERM` stops the loop and the watchdog together. The watchdog still starts before the model server.

## Stopped instances

A stopped Vast.ai instance still bills for storage, and its watchdog is not running. So where a stop leaves a bill behind, the policy destroys instead of stopping, for idle, `max_lifetime` and the budget alike. `lifecycle.keep_disk: true` keeps the old behavior (stop, restart in about a minute) for users who accept the storage bill.

A stopped instance is then destroyed once it has been stopped longer than `lifecycle.stopped_ttl` (reason `stopped_ttl`). This needs destroy support and no active lock. Nothing runs on a stopped machine, so only your computer can enforce it: `gpu guard`, `gpu sweep`, or the timer `gpu sweep --install-timer` writes (systemd on Linux, a LaunchAgent on macOS). `gpu agent install` asks once whether to install that timer. A manual `gpu stop` on Vast.ai is covered by the same rule.

A manual stop is the easy one to forget, so it is never silent. Where a stopped machine still bills, `gpu stop` (and the MCP `gpu_stop`) returns `still_billing: true`, the disk's cost per day (`stopped_cost_usd_per_day`, from Vast's `storage_cost` and the disk size) and a `warning` that says whether the sweep timer will remove it after `stopped_ttl`. After that, every `gpu` command, in any directory, warns on stderr about each registered machine still stopped and billing, until the timer is on (or, with it, once one is past its `stopped_ttl`). That check reads state files only; it never calls the provider. `GPU_NO_STOPPED_CHECK=1` silences it. The skill tells the agent to destroy, not stop, once the work is done and its results are home.

`stopped_since` keeps the first time the CLI saw the instance stopped, but only while nothing says it ran since. A session start, local activity, watchdog contact or a provider start time newer than `stopped_since` means the instance was restarted and stopped again while the CLI was not looking (from the web console, say), so the clock restarts at the current observation instead of destroying a fresh stop.

## Job results

A job's results are on the machine's disk until someone downloads them. No automatic shutdown deletes them unseen.

On the machine, the job runner marks a job's outputs saved once `GET /jobs/<id>/outputs` streamed them after the job ended and nothing changed since. While any are unsaved, it changes what the watchdog does (a `TERMINATE_HOOKS` entry): a destroy becomes a stop where a stop keeps the disk (`outputs_kept`); where it would not (no persistent disk, or no stop at all), an idle shutdown waits up to 2 hours after the last job ended (`outputs_hold`). After that, or when `max_lifetime` or the budget fires there, the machine goes and the event is `outputs_lost`. An explicit `gpu stop` or `gpu destroy` and the stop-failure escalation are not changed.

On this computer, `gpu guard` and `gpu sweep` download the results of finished jobs into `gpu-outputs/<job id>/` while the machine runs, and before any automatic destroy. A stopped machine that still holds results, on a provider where a stopped machine bills and without `keep_disk`, is started, emptied and destroyed (reason `job_outputs_saved`); a failed attempt is retried after an hour and recorded as `job_outputs_kept`. `gpu job pull` starts a stopped machine too, and puts it back afterwards. `gpu destroy` downloads first and refuses when it cannot, unless `--discard-outputs`. `gpu status` lists the jobs whose results are only on the machine (`job_outputs_unsaved`).

## Queue mode: a deadline, not a lifetime

A pool machine has no fixed age. The coordinator gives its watchdog a **deadline** (`POST /deadline`, never moved backwards) that covers the job it just took, that job's `timeout` plus `queue.defaults.deadline_margin`, and moves it again with each job. The watchdog ends the machine at the deadline even with work running (reason `deadline`, which a lock does not override), or after `idle_timeout` with nothing running, whichever comes first. So a machine whose coordinator died finishes what it was given and goes; a warm machine with work keeps going. The deadline applies only when the coordinator sets one: a project without `queue:` has none.

When a machine ends, the same rule as for single jobs applies: outputs not yet downloaded are fetched first, or kept on a stopped disk, and `gpu sweep` saves them before any automatic destroy. See [QUEUE.md](../QUEUE.md).

## A second path

Run the same rule from anywhere that has API access:

```bash
*/5 * * * * cd /path/to/experiment && gpu guard
```

`guard` evaluates the policy against reconciled state and enforces it. `gpu sweep --install-timer` runs it for every registered project; see [Keeping the account clean](USAGE.md#keeping-the-account-clean).

Inference traffic is counted on the instance only. When the watchdog is enabled and installed but did not answer during this reconcile, `guard` therefore does not apply `idle_timeout`: its local idle clock would stop an endpoint that is busy serving requests. `max_lifetime`, the session budget and `stopped_ttl` still apply, and the decision detail records `idle_skipped: watchdog_unreachable`. With `lifecycle.watchdog.enabled: false` the CLI guard owns the idle rule as before.

## A machine that vanished

A destroyed machine leaves nothing to ask: the provider forgets it, and its watchdog went with it. When the CLI next finds the machine missing, it replays this project's own rules (`idle_timeout`, `max_lifetime`, the session budget) on what it last knew. If they called for a destroy by now (on a provider where a stop bills and without `keep_disk`), it records `instance_destroyed` with that `reason`, `by: watchdog` and `inferred: true`, and `state_reconciled` carries `likely_reason`, e.g. `idle_timeout (watchdog)`. Otherwise `likely_reason` is `unknown`: the provider console, another tool, the host, or an account out of credit.

The watchdog's own events (`watchdog_identified`, `instance_idle`, ...) used to reach the local log only on a later command. `gpu setup` now fetches them as soon as the machine is ready, so a machine that goes while nobody runs gpu still has the record that it was guarded.

## A model server that dies at boot

The runtime start command writes the model server's pid to `/var/run/jambu/vllm.pid`. `GET /state` reports `runtime_alive`: `true` or `false` from that pid, `null` while there is no pid file yet (or only one older than the current container start). If the process is gone while the instance is still starting, the watchdog logs `runtime_exited` once and ends startup, so the idle rules apply instead of a startup grace (`boot_timeout` + `startup_timeout` + 5 minutes) kept open by a dead server.

`gpu setup` asks the watchdog on every readiness poll. When `runtime_alive` is `false` it fails at once with the last 30 lines of the model server's log in the error (a CUDA driver mismatch, an out-of-memory, a bad flag), and `cleanup_on_setup_failure` destroys the instance.

`GET /logs?name=runtime|onstart|watchdog&lines=N` (authenticated, default 200, at most 2000) serves the tail of those files without SSH. `gpu logs` reads it.

## The watchdog credential

The watchdog stops its own machine, so it needs provider API access. With the built-in provider (Vast.ai), `VAST_API_KEY` is written on the instance at `/etc/jambu/watchdog.json` (mode 600).

Use a dedicated, restricted Vast.ai key for this, not a personal key. Set `lifecycle.watchdog.enabled: false` to opt out. Shutdown then depends on `gpu guard` or on the CLI staying alive, and `validate` warns you.

## Architecture

```
gpu.yaml (gpu_runtime:)
    │
    ▼
gpu
    │
    ├── Lifecycle    core/policy.py     the only shutdown rule
    ├── Runtime      runtime/vllm.py    how the model runs
    └── State        core/state.py      .gpu/state.json
                │
        Provider contract   providers/base.py
                │
        ┌───────┴────────┐
        ▼                ▼
    Vast adapter    future adapter
```

Four concerns stay independent: model, inference runtime, lifecycle policy, infrastructure provider. The core never branches on a provider name. It branches on `provider.capabilities`.

```
gpu4agents/
├── cli/          setup, run, status, lifecycle, providers, validate
├── core/         config, models, policy, state, events, engine, execution, health
├── providers/    base, registry, vast/
├── runtime/      base, vllm
└── watchdog/     builder, agent_body
```

Adding a provider: [ADDING_A_PROVIDER.md](ADDING_A_PROVIDER.md). The guarantees and the tests that hold them: [ACCEPTANCE.md](ACCEPTANCE.md).
