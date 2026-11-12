# Queue mode: jobs on a pool of GPUs

Agent mode rents one machine for one conversation. Queue mode is for an application that sends jobs over the day: you submit a job directory, a coordinator opens and closes machines as the queue grows and empties, runs the job on one of them, runs it again somewhere else if the machine dies, and keeps the outputs. Any GPU provider works; the pool is rented from the same `gpu.yaml`, through the same provider contract.

A project without a `queue:` section behaves exactly as before. Writing a job is unchanged: it is a directory with `gpujob.yaml`, as [jobs.md](https://gpu4agents.jambu.ai/jobs.md) describes. It gains two optional fields, `max_cost_usd` and `min_vram_gb`.

Two guarantees from agent mode carry over. A machine with nothing to do for `idle_timeout` (15 minutes) ends, by the same rule as today. And the coordinator is allowed to die: every machine keeps its own watchdog and a hard deadline, so a pool with no coordinator drains to zero by itself.

## Contents

- [Quickstart](#quickstart)
- [The pieces](#the-pieces)
- [Configuration](#configuration)
- [Credentials](#credentials)
- [Submitting and following jobs](#submitting-and-following-jobs)
- [How a job runs](#how-a-job-runs)
- [Controls: pause, resume, drain](#controls-pause-resume-drain)
- [Running the coordinator](#running-the-coordinator)
- [Observing it: events, metrics, the auditor](#observing-it)
- [Alerts](#alerts)
- [What can go wrong](#what-can-go-wrong)
- [Limits of this version](#limits-of-this-version)

## Quickstart

One machine, one host, nothing to install beyond gpu4agents. The local queue is a SQLite file in the project, and outputs land in `gpu-outputs/`.

```yaml
# gpu.yaml
gpu_runtime:
  version: 1
  model: {id: Qwen/Qwen2.5-7B-Instruct}   # or any model; jobs may use the server over loopback
  compute: {gpu: {min_vram_gb: 24}}
  queue:
    max_machines: 2
    max_usd_per_hour: 2.00
```

```bash
gpu queue submit ./my-job            # prints the job id at once; nothing is rented yet
gpu queue run                        # the coordinator: opens machines, runs jobs (leave it running)
gpu queue status                     # queue counts, pool machines and what they cost per hour
gpu queue status JOB_ID
gpu queue pull JOB_ID                # verified outputs into gpu-outputs/JOB_ID/
```

Starting `gpu queue run` rents GPUs and spends money. Decide the caps first (`max_machines`, `max_usd_per_hour`, and each job's `timeout` and `max_cost_usd`).

## The pieces

| Piece | What it is |
| --- | --- |
| Queue | Where jobs wait. Behind a backend contract. Built in: `local` (SQLite, one host) and `redis` (several hosts share it). Any other queue plugs in through [the contract](https://github.com/jambuai/gpu4agents/blob/main/docs/QUEUE_BACKENDS.md). |
| Output store | Where job bundles and results live. Built in: `local` (a directory) and `s3` (any S3-compatible bucket). Plugs in the same way. |
| Coordinator | `gpu queue run`. One process per queue holds a lease; a second one waits as a standby and takes over when the lease expires. Any number of processes may submit. |
| Pool | The machines, one per slot, labeled `jambu-<name>-<slot>-<fp6>`. They grow when jobs wait and shrink when machines have nothing to do. |
| Preflight | Runs when a job is claimed. Sets the job's limits (time, cost) and refuses a job that cannot run, with the reason. |
| Auditor | Checks the queue's guarantees (invariants I1 to I8) and repairs what it safely can. |

## Configuration

`gpu_runtime.queue` turns queue mode on. Every field and its default is in [docs/CONFIG.md](https://github.com/jambuai/gpu4agents/blob/main/docs/CONFIG.md#queue). The ones to decide first:

| Field | Default | What it controls |
| --- | --- | --- |
| `backend` | `local` | `local`, `redis`, or `module:factory` for your own |
| `outputs` | `local` | `local`, `s3`, or `module:factory` |
| `namespace` | none | The name of the keyspace on a shared backend |
| `min_machines` | `0` | Machines kept on with an empty queue (they still time out when idle, see [Controls](#controls-pause-resume-drain)) |
| `max_machines` | `1` | The most machines at once |
| `max_usd_per_hour` | none | Cap on what the whole pool bills per hour; blocks opening new machines |
| `scale_up_after` | `30s` | A machine is opened when the oldest queued job has waited this long |
| `jobs_per_machine` | `1` | Jobs a machine runs at once |
| `max_attempts` | `3` | Runs per job before it is failed with every attempt's reason |
| `unresponsive_after` | `5m` | How long a silent machine is waited for before it is destroyed and its jobs requeued |
| `max_machine_lifetime` | none | Optional hard ceiling on a machine's age (a warm machine is otherwise never replaced just for being old) |
| `health_check` | built in | Checked before a machine's first job and between jobs |
| `defaults` | `timeout: 1h` | Limits for a job that sets none |

Set `queue.namespace` when you use Redis. Without it the keyspace is named after the config fingerprint, so it changes when the runtime part of `gpu.yaml` changes, and the new coordinator would look at an empty queue. A name you chose (or the `GPU_QUEUE_NAMESPACE` environment variable) keeps the same keyspace across edits. It is not a secret.

The health check is the GPU being visible and a small CUDA operation succeeding. A project may set its own `health_check.command`. That command must not write job outputs: anything it writes under the job outputs counts as activity and holds the idle shutdown.

## Credentials

Addresses and keys never go in `gpu.yaml` (a URL there is a config error). They come from the environment or `~/.gpu/credentials`, like provider keys:

| Name | Used for |
| --- | --- |
| `GPU_QUEUE_URL` | the Redis queue (`redis://` or `rediss://`); logged with the password removed |
| `GPU_OUTPUTS_URL` | the S3 store, `s3://bucket/prefix` |
| `AWS_*` | the S3 store's keys and endpoint (`AWS_ENDPOINT_URL` for a non-AWS store), the standard boto3 names |
| `GPU_QUEUE_NAMESPACE` | optional namespace when `queue.namespace` is not set |
| `GPU_QUEUE_SUBMITTER` | who is recorded as having submitted or controlled (default `user@host`) |

The provider's key never reaches a job, the queue, an output store, an event or a metric. A service started by systemd or launchd does not get your shell's environment, so put the credentials in `~/.gpu/credentials` for it.

The Redis and S3 clients are optional extras: `pip install "gpu4agents[queue-redis]"` and `"gpu4agents[queue-s3]"`. Importing gpu4agents without them works.

## Submitting and following jobs

```bash
gpu queue submit DIR [--key KEY] [--env NAME=VALUE ...]
gpu queue status [JOB]
gpu queue ls [--state STATE]
gpu queue cancel JOB
gpu queue retry JOB
gpu queue pull JOB [--to DIR]
```

- `submit` packs DIR (honouring `exclude`), stores the bundle and returns the job id at once. A bad job (no `run`, a bundle over 2 GB, a `timeout` that does not parse) is rejected before anything is stored. A paused queue still takes the job and says so.
- `--key` makes a submit idempotent: the same key never queues twice, however many replicas retry it.
- Job ids are at most 60 characters. `--env` values reach the job but are redacted (shown as `***`) in every status output and never written to the audit trail or the events.
- Job states: `queued`, `claimed`, `running`, `delivering`, `succeeded`, `failed`, `cancelled`. The last three are final. A job always ends in one of them, with the reason; never "unknown".
- `cancel` is immediate for a queued job. For a running one the coordinator stops it on its machine and keeps the outputs written so far.
- `retry` queues a failed job again; it keeps its attempts.
- `pull` verifies the outputs against their sha256 manifest before it hands them over.

From Python, an application does not need to shell out:

```python
from gpu4agents.queue import api
job_id = api.submit("train/", key="train-2026-11-12")   # idempotent on the key
info = api.wait(job_id, timeout_s=3600)
path = api.outputs(job_id)
```

Agents use the MCP tools `gpu_queue_submit`, `gpu_queue_status`, `gpu_queue_cancel`, `gpu_queue_pull` and `gpu_queue_control`. They never start the coordinator: that rents GPUs and is the user's decision.

## How a job runs

1. The coordinator claims the oldest queued job. The claim is a lease: nobody else sees the job until it expires or the job is settled.
2. The preflight decides the job's limits: its `timeout`, its `max_cost_usd` and `min_vram_gb`, with the project's `queue.defaults` filling what it leaves out. A job that cannot run (it needs more VRAM than the pool's card, or its cost limit is below what its timeout costs at the pool's price) fails at once with the reason and rents nothing. A project can name its own preflight (`queue.preflight: module:function`) to decide per job.
3. A machine that passed the health check takes the job. The coordinator moves that machine's deadline in its watchdog to cover the job's timeout plus a margin. While work keeps arriving the deadline keeps moving, and a warm machine runs job after job without being set up again. If the coordinator dies nobody moves it: the machine finishes what it has and ends at its deadline at the latest, or after `idle_timeout` with nothing running, whichever comes first.
4. When the job ends, the coordinator fetches its outputs, stores them in the output store with a sha256 manifest, verifies them, and only then settles the job as `succeeded`. Nothing on the machine is deleted before that.

When a machine disappears, stops answering for `unresponsive_after`, or fails the health check right after a job, the job goes back to the queue and runs on another machine, up to `max_attempts`. A job that exits non-zero while the machine is healthy is the job's fault and is not retried automatically (`gpu queue retry` does it by hand). Every attempt's outcome is recorded: `succeeded`, `job_failed`, `timeout`, `machine_lost`, `unhealthy`, `unresponsive`, `preflight_rejected`, `cancelled`, `delivery_failed`, `drained`, `not_started`.

A retried attempt starts from scratch, but its `run` is prefixed with `export JAMBU_PREVIOUS_OUTPUT_DIR=...`: the verified outputs of the earlier attempt are there, so a job that writes checkpoints can resume. The job decides whether to use them. The same directory is described in [jobs.md](https://gpu4agents.jambu.ai/jobs.md).

Outputs that could not be stored (the store was down) stay under `<state_dir>/queue/work/` on the coordinator's host and are retried; the auditor reports any that are an hour old (I8). A store outage never retires machines: outputs the machine still holds are kept, not given up on.

## Controls: pause, resume, drain

| Command | What it does |
| --- | --- |
| `gpu queue pause` | Stops new claims. Running jobs go on. Idle machines, `min_machines` included, time out by the normal idle rule, so a paused queue lets the pool fall to zero |
| `gpu queue resume` | Starts jobs again, and drops a drain that was requested |
| `gpu queue drain` | Pauses, and lets the machines idle out and stop |
| `gpu queue drain --now` | Also sends running jobs back to the queue and retires every machine at once, pulling outputs first. Those jobs are requeued, not failed, and a drained attempt does not count toward `max_attempts` |

`drain --now` through MCP needs `confirm=true` and the user's yes. Every control action is recorded in the audit trail with who did it, and a coordinator that holds the lease does the work itself; without one, the command does it directly.

## Running the coordinator

```bash
gpu queue run [--once] [--metrics-port PORT]
gpu queue install-service [--uninstall]
```

`run` stops on Ctrl-C or SIGTERM; machines keep their deadlines. `--once` runs one tick and exits. `--metrics-port` serves `/metrics` on `127.0.0.1` only. `install-service` writes a user service (systemd on Linux, launchd on macOS) for this project and prints the command that enables it; it does not enable it. A systemd timer and a service do not inherit `devrun` or your shell: put credentials in `~/.gpu/credentials`.

The coordinator renews its lease from a heartbeat thread, so a long upload does not lose it. Leases use the backend's clock, never the caller's. The coordinator lease is 15 s at the default tick (3 ticks, never less); a job's lease is 60 s (12 ticks, never less).

A coordinator that takes over (after a crash or an expired lease) looks for every job its predecessor left in flight. A run its machine lists is adopted, never started twice. When the predecessor died after it recorded an attempt but before the machine got the job, the new coordinator asks that machine's job runner: if the runner has no record of the run, no upload in progress and no code waiting for a start, the run never started. Before the job goes back to the queue, the coordinator fences that remote id on the machine: the runner records it as refused and deletes any code it holds for it, so an upload or a start that the old coordinator still sends late gets a 409 and never runs (an old coordinator that sees the refusal drops the job). If the run did start in the meantime, the runner says so and the run is adopted instead. Once the fence is acknowledged, the job goes back to the queue at once with the outcome `not_started`. Like a drained attempt, it does not count toward `max_attempts`, and the next attempt gets a new remote id; a job whose runs fail to start more than twice `max_attempts` times fails with every reason listed. If the machine lists its jobs but the fence does not get through, nothing is requeued early: the job waits as long as an upload could take, as before. If code for the run is still arriving, the coordinator waits for it. While it waits it sends the machine a keepalive every tick, so the machine does not reach `idle_timeout` with that run still undecided.

After a takeover by a host whose pool is different (another computer, a new state directory), a job running on a machine the new coordinator cannot see waits until that attempt's deadline plus a grace before it runs again, because destroying or reaching the machine is the only way to be sure the first run is over.

`gpu sweep` and the sweep timer know about pool machines: with a dead coordinator, a stopped machine still ends in a destroy, with finished jobs' outputs saved first, and a pool machine is never treated as an orphan.

## Observing it

Every decision is an event in `.gpu/logs/events.jsonl`; metrics are in `.gpu/queue/metrics.prom` and on `/metrics`; the auditor checks the guarantees. `gpu queue stats` shows the metrics as of the coordinator's last tick; `gpu queue audit [--repair]` runs the auditor (exit 1 while a violation remains).

| Concern | Events | Metrics | Invariant or alert |
| --- | --- | --- | --- |
| Coordinator | `coordinator_started`, `coordinator_lease_acquired`, `coordinator_lease_lost`, `coordinator_stopped` | `gpu_queue_coordinator_heartbeat_age_seconds`, `gpu_queue_lease_renew_failures_total` | I7 lease fresh |
| Machines | `pool_machine_requested`, `_ready`, `_unhealthy`, `_unresponsive`, `_gone`, `_retired` | `gpu_pool_machines{state}`, `gpu_pool_cold_start_seconds`, `gpu_pool_health_failures_total` | I4 no orphan machine |
| Deadlines | `pool_deadline_extended`, `instance_destroyed reason=deadline` | `gpu_pool_deadline_remaining_seconds{slot}` | remaining below 0 with a job running |
| Retries | `queue_job_retried`, `queue_job_dead` | `gpu_queue_attempts_total{outcome}`, `gpu_queue_retries_total{cause}` | retry rate |
| Ownership | `queue_job_claimed`, `queue_lease_expired`, `queue_stale_write_refused` | `gpu_queue_stale_writes_total` | I1 every running job is on a live machine; I2 no job on two machines |
| State | `queue_invariant_violated` | `gpu_queue_invariant_violations_total{code}` | any violation |
| Outputs | `queue_outputs_stored`, `queue_outputs_delivery_failed`, `queue_outputs_lost` | `gpu_queue_outputs_pending`, `gpu_queue_outputs_oldest_pending_seconds`, `gpu_queue_outputs_bytes_total` | I3 manifests verify; I8 outputs stored |
| Latency | `queue_job_submitted`, `queue_job_started`, `queue_job_succeeded`, `queue_job_failed` | `gpu_queue_depth`, `gpu_queue_oldest_queued_seconds`, `gpu_queue_job_wait_seconds`, `gpu_queue_job_run_seconds` | I6 no job past its deadline plus grace |
| Money | `pool_scale_decision`, `pool_cap_blocked`, `queue_preflight_rejected` | `gpu_pool_usd_per_hour`, `gpu_pool_spend_usd_total`, `gpu_queue_job_cost_usd` | I5 spend within `max_usd_per_hour` |
| Who and what | `queue_job_submitted` (submitted_by), `queue_control` (action, actor) | `gpu_queue_jobs_total{final_state}` | the audit trail is append-only |
| Control | `queue_control` | `gpu_queue_paused` | pause honored within one tick |

The invariants: I1 running job without a machine, I2 duplicate execution, I3 outputs unverified, I4 orphan machine, I5 spend over the cap, I6 job past its deadline, I7 stale coordinator, I8 outputs not stored.

## Alerts

Page someone when any of these holds:

- the coordinator heartbeat age is more than 3 ticks;
- any invariant violation (`gpu_queue_invariant_violations_total` rises, or `gpu queue audit` exits 1);
- the oldest pending output delivery is older than 10 minutes;
- the retry rate is above 20% over 1 hour (`gpu_queue_retries_total` leaves out attempts sent back by `drain --now` and runs that never started, which are not failures);
- the pool's spend is at or above 90% of `max_usd_per_hour`;
- the oldest queued job is older than 2 × `scale_up_after` plus the cold start p50 while `blocked_by` is set (the last `pool_scale_decision` says which cap: `max_machines`, `max_usd_per_hour`, `no_offer` or `backoff`).

## What can go wrong

| Symptom | What it means |
| --- | --- |
| Jobs stay queued and `blocked_by` is `no_offer` or `backoff` | The provider has no matching capacity. Jobs wait, they do not fail, and the opening of machines backs off up to 5 minutes |
| A job is `delivering` for a long time | The output store is unreachable or the manifest does not verify. Nothing is marked delivered; it retries. The machine's outputs are held |
| The queue looks empty after a `gpu.yaml` edit | The Redis keyspace followed the config fingerprint. Set `queue.namespace` |
| `gpu queue status` says no coordinator holds the lease | Start `gpu queue run` (or enable the service). Machines already running end by their deadlines |
| Exit 4 from a `gpu queue` command | The queue or its output store is unreachable; the message names it, without the password |

## Limits of this version

- Redis must be 5 or newer and a single node. Redis Cluster is not supported (its atomic steps are Lua scripts over several keys).
- One pool per project: one `gpu.yaml`, one provider, one kind of card. No priorities, schedules or job dependencies.
- It is for batch work, not for autoscaling a live model endpoint.
- There is no Jambu server. The queue and the store are ones you run or rent.
