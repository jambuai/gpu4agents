# Queue mode: production jobs on a pool of GPUs

Status: design, awaiting approval. Date: 2026-11-12.

## Why

gpu4agents was built for agents and experiments: one conversation rents one
machine, uses it and gives it back. The team is preparing to use it in
production, where jobs arrive from an application over the day, not from a
chat. Four pieces are missing for that:

1. open and close several GPUs automatically as the queue grows and empties;
2. check that a machine is healthy before (and while) it takes work;
3. run a job again on another machine when the first one disappears;
4. process jobs back to back on a GPU that is already on.

Shutting a machine down after 15 minutes without work stays the rule: it is
what keeps an idle GPU from billing. Queue mode builds on it rather than
replacing it.

No incident triggered this. The team is getting ready for the scenario, so
the design must not assume one workload: jobs may take seconds or hours, and
loading the model may be expensive. Because of that, nothing in queue mode is
a fixed number: limits are configured per job, and a warm machine is never
thrown away just to reset a counter.

## Goals

- A **queue** per project, behind a **queue backend contract**. Jobs are
  submitted to the queue, not to a machine. A job is exactly what JOBS.md
  describes today (a directory with `gpujob.yaml`, outputs in
  `$JAMBU_OUTPUT_DIR`); nothing about writing a job changes.
- Two backends built in: a **local** one (a file on the coordinator's host,
  no server) and a **remote** one, so several application replicas on
  different hosts can share one queue. Any other queue (SQS, a company broker,
  a database the team already runs) plugs in by implementing the contract,
  the same way a GPU provider plugs in through an adapter today.
- A **pool** of machines per project, all rented from the same `gpu.yaml`.
  The pool grows when jobs wait and shrinks when machines have nothing to do.
- A **preflight** per job that sets how that job runs: its time and cost
  limits, and therefore how long the machine running it may live. The limits
  come from the job and the project's queue policy, not from fixed defaults.
- **Health**: a machine receives jobs only after passing a check, and leaves
  the pool when a later check fails.
- **Retry**: a job whose machine vanished, stopped answering or failed its
  health check goes back to the queue and runs on another machine, up to a
  limit; after that it is recorded as failed with every attempt's reason.
- **Batch**: a machine that finishes a job takes the next one without being
  set up again. The HF cache on its disk is reused, and in a project that
  serves a model the model stays loaded between jobs (jobs reach it over
  loopback, as JOBS.md section 1 already allows). A warm machine keeps
  working as long as there is work and its limits allow; there is no
  scheduled replacement.
- Every decision (machine opened, closed, marked unhealthy, limits extended,
  job retried) is an event in `.gpu/logs/events.jsonl`, like everything else
  today.

## Non-goals

- A hosted service. There is no Jambu server; the remote queue is one the user
  runs or rents, reached with the user's own credentials.
- Sharing one pool between projects.
- Mixing GPU types or providers inside one pool (one `gpu.yaml`, one pool).
- Priorities, schedules, job dependencies or DAGs.
- Autoscaling a model endpoint for live request traffic (the OpenAI-compatible
  server stays the agent-mode feature it is today).

## Anti-goals (failure even if it technically works)

- **A GPU that bills while idle.** Whatever the coordinator does, a machine
  with no work for `idle_timeout` must end, with the same rules as today
  (destroy on Vast.ai, stop where a stop ends billing, outputs kept until
  downloaded).
- **A coordinator crash that leaves machines running.** The coordinator is a
  client process and will die (deploys, reboots, OOM). Each machine keeps its
  own watchdog and a hard deadline (see "Limits"), so with the coordinator
  gone the pool drains to zero by itself.
- **A cold start that was not needed.** Replacing a healthy, warm machine only
  because it reached a fixed age or spend is a failure of this design.
- **A job lost silently.** Every submitted job ends in a final state the
  client can read: succeeded, failed (with the reason and attempts), or
  cancelled. Never "unknown".
- **A job run twice at the same time** because two machines both think they
  own it.
- **Outputs deleted before they are downloaded**, the guarantee 0.12.0 made
  for single jobs.
- **Spending past the caps** the user set, even while catching up on a long
  queue.
- **A queue contract so narrow that a real broker cannot implement it, or so
  loose that the built-ins behave differently.** Both built-ins pass the same
  contract test suite, and that suite is what a third-party backend runs too.

## Constraints

- Provider-agnostic: only the existing provider contract (`providers/base.py`)
  plus the watchdog. No provider-specific queueing.
- Secrets as today: the provider key never goes to the job; no key is printed.
  The remote queue's credentials follow the same rules as provider keys
  (`gpu login`-style storage or environment, never in `gpu.yaml`).
- Agent mode keeps working unchanged. A project without a `queue:` section
  behaves exactly as 0.13.0, including its 6 h / budget defaults.
- Python 3.10+. The local backend needs nothing installed. The remote
  backend's client library is an optional extra (`pip install
  gpu4agents[queue-redis]`), not a core dependency.

## Decided

Decided by the user:

- **The coordinator runs in each client that uses gpu4agents**, not on a
  Jambu server and not on the GPUs.
- **Queue backends are pluggable**, with a strong but flexible contract: a
  local backend and a remote backend ship built in, and any other queue can be
  used by implementing the contract.
- **Limits are not fixed.** A preflight configures, per job, the approach that
  fits it (time and cost limits). The 6 h / $5 agent-mode defaults do not
  bind queue mode.
- **No replacement of a warm machine to reset limits.** If the structure is
  up, it keeps working; adequate configuration is how long work is allowed.
- The 15-minute idle shutdown stays the base behavior; queue mode is a mode on
  top of it.
- The job contract (JOBS.md) is the unit of work.

Proposed by Claude in the conversation and accepted when the user said to go
ahead:

- `min_machines` (default 0): machines kept on even with an empty queue.
- `max_machines`: the most machines the pool may have at once.
- A spend cap per hour for the whole pool (`max_usd_per_hour`), on top of the
  per-job cost limits.
- A new machine is opened when the oldest queued job has waited longer than
  `scale_up_after`, within `max_machines` and the spend cap.
- A health check before a machine's first job and between jobs; a machine
  that fails leaves the pool, is destroyed (after its outputs are downloaded,
  when they can be) and its host is marked bad, as `setup` does today.
- At most `max_attempts` runs per job (default 3), then failed with every
  attempt's reason.

Claude's calls after the design review (check these):

- **Remote backend: Redis.** Chosen because it is the most common queue the
  team is likely to already have or can rent managed (Upstash, ElastiCache,
  Redis Cloud), it has atomic claims with expiry, and its client is small.
  Postgres (`SELECT ... FOR UPDATE SKIP LOCKED`) is the natural second
  third-party backend and the first real test of the contract.
- **A stuck machine** (running, but its watchdog stops answering): the
  coordinator waits a grace period (`unresponsive_after`, default 5 min),
  tries once to download the outputs, then destroys it and re-queues its jobs.
  Destroying it is the only way to be sure the job is not still running when
  it starts elsewhere. What could not be downloaded is lost and recorded as
  such in the attempt.
- **Retries start from scratch**, with the previous attempt's downloaded
  outputs (if any) made available to the new attempt under
  `$JAMBU_PREVIOUS_OUTPUT_DIR`, so a job that writes checkpoints can resume
  on its own. The job decides whether to use them.
- **Machine fault vs job fault**: a job that exits non-zero is the job's
  fault, unless the health check run right after it fails; then it counts as
  the machine's fault and the job is retried without using up an attempt of
  its own budget (it still counts toward `max_attempts`, to stop bad hardware
  from looping forever).
- **The health check is configurable** in `queue.health_check` (named so it
  does not collide with the existing `health:` section): the default checks
  that the GPU is visible and a small CUDA operation succeeds; the free-memory
  threshold is off by default (a project serving a model holds most of the
  memory in vLLM); a project can supply its own test command.
- **The hourly spend cap counts everything the pool bills**: running
  machines, stopped machines still billing their disk, and machines held
  while their outputs wait for download. It only blocks opening new machines;
  it never stops a machine with a job running. A price rise therefore delays
  growth, it does not kill work.
- **Outputs**: the coordinator downloads them as today and the job's final
  record in the queue carries where they are. The built-in destination is a
  local directory; an output destination is a second small contract (put a
  job's files, return a location), with S3-compatible storage built in, so an
  application replica on another host can read results the coordinator
  fetched.
- **One active coordinator per queue.** The backend contract includes a
  lease; a second coordinator on the same queue waits as a standby and takes
  over when the lease expires. Any number of processes may submit.

## Limits: the preflight

Agent mode keeps its per-machine `max_lifetime` and session budget. Queue mode
replaces them with limits that follow the work:

- Each job carries its own limits: `timeout` (already in `gpujob.yaml`) and a
  new optional `max_cost_usd`. Missing values come from the project's
  `queue.defaults`.
- The **preflight** runs when a job is claimed, before it is sent to a
  machine. It produces the job's effective limits and decides whether the job
  can run at all (for example, refuse a job whose declared VRAM need is above
  the pool's card, or whose cost limit is below what its timeout would cost on
  the pool's price). The built-in preflight applies the rules above; a project
  can provide its own (a Python callable named in `gpu.yaml`) to choose
  differently per job, e.g. from the input size.
- A machine has a **deadline**, not a fixed lifetime: when it takes a job, the
  coordinator extends the machine's deadline in its watchdog to cover that
  job's timeout plus a margin. While work keeps arriving, the deadline keeps
  moving; the machine stays warm for as long as there is work.
- If the coordinator dies, nobody extends the deadline. The machine finishes
  what it was given and ends at its deadline at the latest, or after
  `idle_timeout` with nothing running, whichever comes first. That is what
  keeps the "coordinator crash" anti-goal true without a fixed age limit.
- A hard ceiling per machine (`queue.max_machine_lifetime`, default off) and
  the pool's hourly cap remain for users who want them.

## The queue backend contract (shape)

The builder decides the exact signatures; it must cover:

- `submit(job) -> id`, idempotent on a client-supplied key.
- `claim(worker, lease) -> job | None`: atomic; a claimed job is invisible to
  other claims until its lease expires or it is settled.
- `renew(id, lease)`, `complete(id, result)`, `fail(id, reason, retry: bool)`,
  `cancel(id)`.
- `get(id)`, `list(filter)`, and the counts the scaler needs (queued, running,
  age of the oldest queued job).
- The coordinator lease (acquire, renew, release).
- Job payloads stay small: the bundle (the job directory, packed at submit
  time) is stored through the output-destination contract or the local disk,
  and the queue holds a reference to it.

A contract test suite, run against the local and Redis backends, defines
correct behavior (no double claim, lease expiry re-queues, idempotent submit,
final states). A third-party backend is "supported" when it passes it.

## Other shape (Claude's proposal, open to the builder)

- Configuration: a `queue:` section in `gpu.yaml` (`backend`, `min_machines`,
  `max_machines`, `max_usd_per_hour`, `scale_up_after`, `jobs_per_machine`,
  `max_attempts`, `unresponsive_after`, `health_check`, `defaults`,
  `preflight`, `outputs`). Its presence turns queue mode on.
- Commands: `gpu queue submit [DIR]` (returns a job id at once), `gpu queue
  status [JOB]`, `gpu queue ls`, `gpu queue cancel JOB`, `gpu queue pull JOB`,
  and `gpu queue run`, the coordinator loop, meant to run as a long-lived
  process (systemd unit, container). Matching MCP tools and a Python API
  (`gpu4agents.queue.submit(...)`) so an application need not shell out.
- Per-machine state: one record per pool machine (instance, watchdog token,
  model key, health, deadline, current jobs). Agent mode keeps its
  single-machine `state.json`.
- `gpu sweep` and the sweep timer know about pool machines, so a dead
  coordinator plus a stopped machine still ends in a destroy.

## Left to the builder

- The exact scaling algorithm (how many machines to open at once, cool-down
  between decisions), within the caps and the anti-goals.
- The `jobs_per_machine` default (1 is the safe choice, since jobs declare no
  memory needs).
- Heartbeat and lease durations, and the deadline margin.
- Exact signatures of the backend and output-destination contracts.
- Test strategy: the FakeProvider extended to a pool, the contract suite
  against both backends (Redis in a container in CI), plus one real run on
  Vast.ai with the user's approval before renting.
- Docs: a new `QUEUE.md` (served at `/queue.md`), a guide for writing a queue
  backend (like `providers.md`), a docs-site section, the CHANGELOG, and
  acceptance guarantees in `docs/ACCEPTANCE.md`.
