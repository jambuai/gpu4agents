# Queue mode field run on Vast.ai (2026-11-12)

The plan's Task 16: a real run of queue mode, with faults injected on purpose. Project: `examples/queue-smoke/` (jobs-only machines, `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, 16 GB minimum, `max_machines: 2`, `max_usd_per_hour: 0.80`, `idle_timeout: 5m`). Each job runs CUDA matmuls for 60 to 120 s and writes `result.json`.

## Run 1: 20 jobs, two faults

| Time (UTC) | What happened |
| --- | --- |
| 19:48 | 20 jobs submitted; coordinator 1 started. Two machines opened (RTX 2080 Ti). |
| 19:59 | s1's image pull made no progress for 10 min: host marked bad, destroyed, replaced by an RTX 3080 on another host. No action from us. |
| 20:01 | Fault 1: s2's instance destroyed at the provider while it held a job. The coordinator saw it gone, retired the slot without marking the host bad, requeued the job as `machine_lost`, and opened a new s2. |
| 20:02 | Fault 2: coordinator 1 killed with SIGKILL, one job claimed and recorded but not yet dispatched. |
| 20:03 | Coordinator 2 started after the leases expired; it recovered the in-flight job and resumed. |
| 20:22 | 20 of 20 jobs succeeded. `gpu queue audit`: no violations. 20 verified output directories. |

Pool spend for the whole run, as recorded by the ledger: about $0.13.

## Run 2: coordinator killed and not restarted

Coordinator 2 killed with SIGKILL at 20:24, both machines idle. With no coordinator, each machine's watchdog ended it at `idle_timeout`: two machines at 20:27, one at 20:28, none at 20:29 (`gpu ls`).

## What the run found, and what changed

The crash at 20:02 landed between the coordinator recording an attempt and sending it to the machine. The next coordinator could not tell whether the upload had started, so it waited the upload timeout (11 min); meanwhile the idle rule ended the assigned machine and the job spent an attempt. Nothing ran twice and nothing was lost, but it was wasteful. Fixed in the same branch:

- the machine's job runner lists uploads and bundles that never started;
- a coordinator that takes over fences a run that never started on its home machine (`POST /jobs/<id>/refuse`), then requeues it at once as `not_started`, which does not use an attempt;
- a machine with a pending decision is kept awake;
- a machine restarted to pull outputs starts with a fresh deadline.

Reproduce: `cd examples/queue-smoke`, submit with `gpu queue submit job --key smoke-N --env SMOKE_SECONDS=60`, run `gpu queue run`, follow `gpu queue status`.

## Run 3: the final code (21:45 to 22:05 UTC)

Ten more jobs on the code with every fix in. Six ran to success after a coordinator restart; then four more, with the coordinator killed by SIGKILL while two of them were running on warm machines. The new coordinator adopted both running jobs (no retry, no second start) and finished all four. 30 of 30 jobs succeeded across the three runs, `gpu queue audit` found no violations, and with the coordinator stopped both machines ended by themselves within the idle timeout (none listed at 22:05).
