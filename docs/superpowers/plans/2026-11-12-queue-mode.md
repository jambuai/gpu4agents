# Queue Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Production queue mode for gpu4agents: a pluggable job queue (local and Redis built in), a pool of GPU machines that grows and shrinks with it, per-job limits set by a preflight, health checks, retries on another machine, and outputs delivered to a pluggable destination (local and S3-compatible built in), all observable and controllable.

**Architecture:** A new package `gpu4agents/queue/` holds the contracts (queue backend, output store), their built-in implementations, the preflight, a pure scaling function, the pool (one `Engine` per machine, each with its own state directory) and the coordinator loop that runs in the client. Machines keep their own watchdog; queue mode replaces `max_lifetime`/session budget with a per-machine deadline the coordinator keeps extending, so a dead coordinator lets the pool drain by itself. Every decision goes through one `Observer` (events + metrics), and an auditor checks the invariants that define "nothing lost, nothing run twice, nothing billed for nothing".

**Tech Stack:** Python 3.10+, pydantic, typer, httpx, sqlite3 (stdlib), `redis>=5` (extra `queue-redis`), `boto3>=1.34` (extra `queue-s3`), pytest, fakeredis (dev), moto (dev).

**Spec:** `docs/superpowers/specs/2026-11-12-queue-mode-design.md`

## Global Constraints

- A `gpu.yaml` without `gpu_runtime.queue` behaves exactly as 0.13.0 (all existing tests pass unchanged).
- The job contract is JOBS.md unchanged; `gpujob.yaml` only gains the optional `max_cost_usd` and `min_vram_gb`.
- No provider key reaches a job, the queue, an output store, an event or a metric. Queue/S3 credentials come from the environment or `~/.gpu/credentials` (`GPU_QUEUE_URL`, `GPU_OUTPUTS_URL`, standard `AWS_*`), never from `gpu.yaml`, and are never printed (URLs are logged with the password removed).
- Core install gains no dependency: `redis` and `boto3` are optional extras; importing `gpu4agents` without them works.
- Defaults from the spec: `min_machines: 0`, `max_attempts: 3`, `unresponsive_after: 5m`, `max_machine_lifetime: null`, `idle_timeout` stays 15m, `jobs_per_machine: 1`.
- Machines are labeled `jambu-<name>-<slot>-<fp6>` (fingerprint stays the last segment, so `gpu adopt` and `gpu sweep` keep working).
- Commit messages: one plain sentence of what changed, no AI attribution trailers.
- Version 0.14.0 (pyproject, `__init__`, both plugin.json).

## Observation points (resilience, consistency, delivery, governance, control)

Each point is an event in `.gpu/logs/events.jsonl` (via `Observer.emit`), a metric in `.gpu/queue/metrics.prom` (via `Observer.inc/gauge/timing`) and, where it is a guarantee, an invariant the auditor checks (Task 12). The tasks that produce them are named; QUEUE.md (Task 15) documents the alert thresholds.

| Concern | What is observed | Event(s) | Metric(s) | Invariant / alert | Task |
|---|---|---|---|---|---|
| Resilience: coordinator | who coordinates, lease renewals, takeovers | `coordinator_started` (queue config hash), `coordinator_lease_acquired`, `coordinator_lease_lost`, `coordinator_stopped` | `gpu_queue_coordinator_heartbeat_age_seconds`, `gpu_queue_lease_renew_failures_total` | I7 lease fresh; alert heartbeat age > 3× tick | 11 |
| Resilience: machines | open, ready, unhealthy, unresponsive, gone, retired, cold start time | `pool_machine_requested`, `pool_machine_ready`, `pool_machine_unhealthy`, `pool_machine_unresponsive`, `pool_machine_gone`, `pool_machine_retired` | `gpu_pool_machines{state}`, `gpu_pool_cold_start_seconds`, `gpu_pool_health_failures_total` | I4 no orphan machine | 10 |
| Resilience: deadlines | every extension and every deadline shutdown | `pool_deadline_extended`, `instance_destroyed reason=deadline` (watchdog) | `gpu_pool_deadline_remaining_seconds{slot}` | alert remaining < 0 with a job running | 1, 10 |
| Resilience: retries | why each attempt ended | `queue_job_retried` (cause), `queue_job_dead` | `gpu_queue_attempts_total{outcome}`, `gpu_queue_retries_total{cause}` | alert retry rate > 20% over 1 h | 11 |
| Consistency: ownership | claim, fencing token, lease expiry, stale writes refused | `queue_job_claimed`, `queue_lease_expired`, `queue_stale_write_refused` | `gpu_queue_stale_writes_total` | I1 every running job is on a live machine; I2 no job runs on two machines | 3, 11, 12 |
| Consistency: state | queue vs machines vs outputs agree | `queue_invariant_violated` (code, job, machine) | `gpu_queue_invariant_violations_total{code}` | any violation is an alert | 12 |
| Delivery: outputs | fetched, stored, verified, failed, lost | `queue_outputs_stored` (files, bytes, sha256 manifest), `queue_outputs_delivery_failed`, `queue_outputs_lost` | `gpu_queue_outputs_pending`, `gpu_queue_outputs_oldest_pending_seconds`, `gpu_queue_outputs_bytes_total` | I3 every succeeded job's manifest verifies; alert oldest pending > 10 min | 6, 11 |
| Delivery: latency | wait, run, total per job | `queue_job_submitted`, `queue_job_started`, `queue_job_succeeded`, `queue_job_failed` | `gpu_queue_depth`, `gpu_queue_oldest_queued_seconds`, `gpu_queue_job_wait_seconds`, `gpu_queue_job_run_seconds` | I6 no non-final job past its deadline + grace | 11 |
| Governance: money | pool spend now and total, caps that blocked growth, preflight refusals | `pool_scale_decision` (inputs and result), `pool_cap_blocked` (which cap), `queue_preflight_rejected` | `gpu_pool_usd_per_hour`, `gpu_pool_spend_usd_total`, `gpu_queue_job_cost_usd` | I5 spend ≤ `max_usd_per_hour`; alert at 90% | 7, 9, 10 |
| Governance: who and what | submitter of each job, actor of each control action, config in force | `queue_job_submitted` (submitted_by), `queue_control` (action, actor), `coordinator_started` (config hash) | `gpu_queue_jobs_total{final_state}` | audit trail is append-only | 3, 13 |
| Control | pause, resume, drain, stop, cancel, retry | `queue_control` | `gpu_queue_paused` | pause honored within one tick | 13 |

## Review Focus

- **Coordinator killed between claim and dispatch, or between dispatch and the queue update.** Expect: the lease expires, the job is claimed again, and the watchdog's 409 on an already-started remote id is treated as "it is running there" (adopt the run), never as a second start. Test in Task 11: `test_crash_after_dispatch_adopts_the_running_job_instead_of_starting_it_twice`.
- **Hosts with skewed clocks sharing the Redis queue.** Expect: leases and ages use the backend clock (`backend.now()`), never `time.time()` of the caller. Test in Task 5: `test_lease_expiry_uses_the_server_clock`.
- **An output download cut halfway.** Expect: nothing marked delivered, outputs stay on the machine (watchdog holds them), delivery retried; a manifest that fails verification is never `succeeded`. Test in Task 6: `test_a_truncated_upload_fails_verification`; Task 11: `test_delivery_failure_keeps_the_job_delivering_and_retries`.
- **The provider has no capacity for a long time.** Expect: jobs stay queued (not failed), `pool_cap_blocked`/`provider_error` events once per change, exponential backoff of open attempts up to 5 min, no tight loop. Test in Task 10: `test_no_capacity_backs_off_and_keeps_jobs_queued`.
- **A bad submission (no `run`, bundle over 2 GB, `timeout` unparseable, queue paused).** Expect: rejected at submit with a clear error and nothing stored, except paused, which queues and says so. Test in Task 13: `test_submit_rejects_bad_jobs_and_stores_nothing`.

---

### Task 1: Pool-ready Engine, the deadline rule and the watchdog route

**Files:**
- Modify: `gpu4agents/core/policy.py` (Snapshot, PolicyConfig, evaluate)
- Modify: `gpu4agents/watchdog/agent_body.py` (snapshot(), dispatch(): `/deadline`, state_payload)
- Modify: `gpu4agents/core/engine.py` (`Engine.__init__`, label at line ~460, `policy_config`, new `set_deadline`)
- Modify: `gpu4agents/core/beacon.py` (`WatchdogClient.set_deadline`)
- Test: `tests/test_policy.py`, `tests/test_watchdog.py`, `tests/test_engine.py`

**Interfaces:**
- Produces: `policy.REASON_DEADLINE = "deadline"`; `Snapshot.deadline_at: Optional[float] = None`; `PolicyConfig.deadline_enforced: bool = False`.
- Produces: watchdog `POST /deadline {"until": float}` → `{"ok": true, "deadline_at": float}`; `GET /state` includes `deadline_at`.
- Produces: `Engine(config, provider=None, echo=None, credentials=None, *, state_dir: Optional[Path] = None, slot: Optional[str] = None)`; `Engine.slot`; `Engine.set_deadline(until: float) -> float`.
- Produces: `WatchdogClient.set_deadline(until: float) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
def test_deadline_ends_a_machine_even_with_work_running():
    cfg = PolicyConfig(deadline_enforced=True, idle_timeout_s=900)
    snap = Snapshot(now=1000, created_at=0, last_activity=999, deadline_at=999,
                    workloads=[Workload(id="j", started_at=0, last_heartbeat=999)])
    d = evaluate(snap, cfg)
    assert d.terminates and d.reason == "deadline"

def test_no_deadline_set_means_only_idle_and_lifetime_apply():
    cfg = PolicyConfig(deadline_enforced=True, idle_timeout_s=900)
    assert evaluate(Snapshot(now=1000, created_at=0, last_activity=999), cfg).reason == "within_policy"

def test_deadline_is_ignored_outside_queue_mode():
    cfg = PolicyConfig(deadline_enforced=False)
    assert not evaluate(Snapshot(now=1000, created_at=0, last_activity=999, deadline_at=1), cfg).terminates
```
In `tests/test_watchdog.py`: `test_deadline_route_sets_and_reports_the_deadline` (POST `/deadline` until=now+600 → 200; GET `/state` → `deadline_at` equal) and `test_deadline_route_refuses_moving_backwards` (second POST with a smaller `until` → 200 with the larger value kept).
In `tests/test_engine.py`: `test_pool_engine_uses_its_own_state_dir_and_slot_label` (Engine with `state_dir=tmp/pool/s1`, `slot="s1"` → `store.dir == tmp/pool/s1`, provisioned label matches `^jambu-.+-s1-[0-9a-f]{6}$`) and `test_queue_mode_policy_drops_lifetime_and_session_budget` (config with a `queue:` section and `slot` set → `policy_config().max_lifetime_s is None`, `.max_session_cost_usd is None`, `.deadline_enforced is True`; without slot the agent-mode values are unchanged).

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_policy.py tests/test_watchdog.py tests/test_engine.py -k "deadline or pool_engine or queue_mode" -v`
Expected: FAIL (`deadline_at` unexpected keyword, 404 on `/deadline`, `state_dir` unexpected keyword).

- [ ] **Step 3: Implement**

In `evaluate`, after the STOPPED/DESTROYED branches and before `max_lifetime`: when `cfg.deadline_enforced and snapshot.deadline_at is not None and snapshot.now > snapshot.deadline_at`, return `_terminal_action(cfg)` with `REASON_DEADLINE`; a lock does not override it. The watchdog keeps `deadline_at` in `_STATE` (persisted by `save_state`, so a respawn keeps it), stores `max(old, new)`, logs `pool_deadline_extended`. `Engine.policy_config` sets `max_lifetime_s = parse_duration(queue.max_machine_lifetime)`, `max_session_cost_usd=None`, `deadline_enforced=True` when `self.slot and config.queue`. Label: `f"{JAMBU_LABEL_PREFIX}{name[:20]}-{slot}-{fp[:6]}"` when slot is set. (`config.queue` arrives in Task 2; guard with `getattr(config, "queue", None)` until then, removed in Task 2.)

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

`git commit -m "A machine can carry a deadline its watchdog enforces even under work, and an Engine can run as one slot of a pool with its own state directory."`

### Task 2: The `queue:` configuration

**Files:**
- Modify: `gpu4agents/core/config.py` (new `QueueHealthCheck`, `QueueDefaults`, `QueueConfig`; `RuntimeConfig.queue: Optional[QueueConfig] = None`)
- Modify: `gpu4agents/core/job.py` (`JobManifest.max_cost_usd: Optional[float]`, `min_vram_gb: Optional[float]`)
- Modify: `gpu.example.yaml` (commented `queue:` block)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
```python
class QueueHealthCheck(_Base):
    command: Optional[str] = None          # None = built-in CUDA check
    min_free_vram_gb: Optional[float] = None
    timeout: str = "2m"
    between_jobs: bool = True
class QueueDefaults(_Base):
    timeout: str = "1h"
    max_cost_usd: Optional[float] = None
    deadline_margin: str = "10m"
class QueueConfig(_Base):
    backend: str = "local"                 # "local" | "redis" | "module:factory"
    outputs: str = "local"                 # "local" | "s3" | "module:factory"
    min_machines: int = 0
    max_machines: int = 1
    max_usd_per_hour: Optional[float] = None
    scale_up_after: str = "30s"
    jobs_per_machine: int = 1
    max_attempts: int = 3
    unresponsive_after: str = "5m"
    max_machine_lifetime: Optional[str] = None
    tick: str = "5s"
    preflight: Optional[str] = None        # "module:function"
    health_check: QueueHealthCheck = QueueHealthCheck()
    defaults: QueueDefaults = QueueDefaults()
```
  plus `*_s` accessors in seconds on `QueueConfig` (`scale_up_after_s`, `unresponsive_after_s`, `tick_s`, `max_machine_lifetime_s`).

- [ ] **Step 1: Write the failing tests**: `test_no_queue_section_means_agent_mode` (`load_config(...).queue is None`), `test_queue_defaults_match_the_spec` (`min_machines == 0`, `max_attempts == 3`, `unresponsive_after_s == 300`, `max_machine_lifetime_s is None`, `jobs_per_machine == 1`), `test_queue_rejects_max_below_min` (`min_machines: 2, max_machines: 1` → ConfigError), `test_queue_backend_url_in_yaml_is_refused` (`backend: "redis://u:p@h"` → ConfigError mentioning `GPU_QUEUE_URL`), `test_job_manifest_accepts_cost_and_vram`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_config.py -k queue -v` → FAIL.
- [ ] **Step 3: Implement** the models above with `_duration` validators; `backend`/`outputs` values containing `://` are refused (credentials never in gpu.yaml). Remove the `getattr` guard from Task 1.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** `"gpu.yaml takes a queue: section (pool size, spend cap, retries, health check, job defaults); its absence keeps agent mode as it was."`

### Task 3: Queue models, the backend contract, its test suite, and the Observer

**Files:**
- Create: `gpu4agents/queue/__init__.py`, `gpu4agents/queue/models.py`, `gpu4agents/queue/backend.py`, `gpu4agents/queue/testing.py`, `gpu4agents/queue/observe.py`
- Modify: `gpu4agents/core/events.py` (queue/pool event constants listed in the Observation table)
- Test: `tests/test_queue_observe.py`, `tests/test_queue_models.py`

**Interfaces:**
- Produces (`models.py`):
```python
class JobState(str, Enum): QUEUED="queued"; CLAIMED="claimed"; RUNNING="running"; DELIVERING="delivering"; SUCCEEDED="succeeded"; FAILED="failed"; CANCELLED="cancelled"
FINAL_STATES: frozenset[JobState]          # SUCCEEDED, FAILED, CANCELLED
class Outcome(str, Enum): SUCCEEDED, JOB_FAILED, TIMEOUT, MACHINE_LOST, UNHEALTHY, UNRESPONSIVE, PREFLIGHT_REJECTED, CANCELLED, DELIVERY_FAILED
@dataclass class Attempt: number: int; machine: Optional[str]; remote_id: str; started_at: float; ended_at: Optional[float]=None; outcome: Optional[Outcome]=None; exit_code: Optional[int]=None; detail: str=""
@dataclass class QueueJob: id: str; key: str; bundle_ref: str; manifest: dict; submitted_at: float; submitted_by: str; state: JobState=JobState.QUEUED; attempts: list[Attempt]; limits: Optional[dict]=None; outputs_ref: Optional[dict]=None; result: Optional[dict]=None; cancel_requested: bool=False; machine: Optional[str]=None
@dataclass class Claim: job: QueueJob; token: str; lease_until: float
@dataclass class QueueStats: by_state: dict[str,int]; oldest_queued_age_s: Optional[float]
class LeaseLost(JambuError)                 # a write with a stale token
```
  `QueueJob.to_dict()/from_dict()` round-trip (JSON-safe).
- Produces (`backend.py`): `class QueueBackend(ABC)` with `name: str` and exactly:
  `now() -> float`; `submit(job) -> QueueJob` (idempotent on `key`, returns the existing job); `claim(worker: str, lease_s: float) -> Optional[Claim]` (oldest QUEUED, skips paused queue and `cancel_requested`); `renew(claim, lease_s) -> Claim` (raises LeaseLost); `update(claim, **fields) -> QueueJob` (raises LeaseLost); `requeue(claim, attempt: Attempt) -> QueueJob`; `settle(claim, state: JobState, result: dict) -> QueueJob` (final; raises ValueError for non-final state); `request_cancel(job_id: str, actor: str) -> QueueJob`; `retry(job_id: str, actor: str) -> QueueJob` (FAILED → QUEUED, keeps attempts); `get(job_id) -> Optional[QueueJob]`; `list(states: Optional[set[JobState]] = None, limit: int = 100) -> list[QueueJob]`; `stats() -> QueueStats`; `acquire_coordinator(owner: str, lease_s: float) -> bool`; `renew_coordinator(owner: str, lease_s: float) -> bool`; `release_coordinator(owner: str) -> None`; `coordinator() -> Optional[dict]` (`owner`, `lease_until`); `set_paused(paused: bool, actor: str) -> None`; `paused() -> bool`.
  Expired claims return to QUEUED on the next `claim`/`stats` call (lazy reaping), and an expired claim's token is dead.
  `def load_backend(spec: str, config: RuntimeConfig) -> QueueBackend` resolves `"local"`, `"redis"`, or `"module:factory"` (factory gets the config).
- Produces (`testing.py`): `class QueueBackendContract` — a pytest mixin. Subclasses provide fixtures `backend` and `clock` (an object with `advance(seconds)` that moves `backend.now()`). Tests in it (each a method): `test_submit_is_idempotent_on_key`, `test_claim_returns_oldest_first`, `test_a_claimed_job_is_not_claimed_again`, `test_lease_expiry_requeues`, `test_a_stale_token_cannot_write` (LeaseLost on update/settle after expiry and re-claim), `test_settle_is_final` (claim after settle returns None for it; settle twice raises LeaseLost), `test_requeue_records_the_attempt`, `test_cancel_of_a_queued_job_is_immediate` (state CANCELLED), `test_cancel_of_a_claimed_job_sets_the_flag`, `test_paused_queue_claims_nothing`, `test_one_coordinator_lease_at_a_time`, `test_coordinator_lease_expires`, `test_stats_counts_and_oldest_age`, `test_concurrent_claims_never_share_a_job` (8 threads × 50 claims over 100 jobs → each job id claimed once).
- Produces (`observe.py`): `class Observer: __init__(events: EventLog, metrics_path: Path)`; `emit(event: str, **fields) -> dict`; `inc(name: str, by: float = 1, **labels)`; `gauge(name: str, value: float, **labels)`; `timing(name: str, seconds: float, **labels)` (exported as `_count`, `_sum`, `_max`); `write() -> None` (Prometheus text format, atomic replace); `snapshot() -> dict`. `NullObserver` for tests. Fields named `url`, `*_url`, `token`, `password`, `secret`, `api_key` are redacted by `emit`.

- [ ] **Step 1: Write the failing tests**: `test_observer_writes_prometheus_text_atomically` (after `inc("gpu_queue_retries_total", cause="machine_lost")` and `write()`, file contains `gpu_queue_retries_total{cause="machine_lost"} 1`), `test_observer_redacts_secrets` (`emit("x", url="redis://u:pw@h")` → stored `url == "redis://u:***@h"`; `token` → `"***"`), `test_queue_job_round_trips_through_dict`, `test_final_states_are_exactly_three`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_observe.py tests/test_queue_models.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** the modules with the interfaces above. `testing.py` imports pytest lazily (inside the module is fine; it is only imported by tests).
- [ ] **Step 4: Run** the same command → PASS.
- [ ] **Step 5: Commit** `"The queue's data model, the backend contract with a reusable conformance suite, and one Observer for queue events and metrics."`

### Task 4: Local backend (SQLite)

**Files:**
- Create: `gpu4agents/queue/backends/__init__.py`, `gpu4agents/queue/backends/local.py`
- Test: `tests/test_queue_backend_local.py`

**Interfaces:**
- Consumes: `QueueBackend`, models, `QueueBackendContract` (Task 3).
- Produces: `LocalBackend(path: Path, clock: Callable[[], float] = time.time)`; default path `<state_dir>/queue/queue.db`.

- [ ] **Step 1: Write the failing test**: `class TestLocalBackend(QueueBackendContract)` with `backend` = `LocalBackend(tmp_path/"q.db", clock=fake.now)` and `clock` = that fake; plus `test_two_processes_share_the_file` (two `LocalBackend` objects on the same file: a job claimed by one is invisible to the other).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_backend_local.py -v` → FAIL.
- [ ] **Step 3: Implement** with `sqlite3` in WAL mode, `BEGIN IMMEDIATE` around claim/renew/update/settle, a `jobs` table (id, key UNIQUE, state, payload JSON, token, lease_until, submitted_at), a `meta` table (coordinator owner/lease, paused), and an append-only `audit` table (ts, actor, action, job_id, detail). Tokens are `secrets.token_hex(16)`.
- [ ] **Step 4: Run** → PASS (all contract tests).
- [ ] **Step 5: Commit** `"Local queue backend on SQLite, passing the backend contract."`

### Task 5: Redis backend

**Files:**
- Create: `gpu4agents/queue/backends/redis.py`
- Modify: `pyproject.toml` (extras `queue-redis = ["redis>=5"]`; dev gains `fakeredis>=2.23`)
- Modify: `.github/workflows/ci.yml` (a `redis:7` service and `GPU_TEST_REDIS_URL=redis://localhost:6379/15` for one job)
- Test: `tests/test_queue_backend_redis.py`

**Interfaces:**
- Produces: `RedisBackend(url: str, namespace: str)`; namespace = `gpu4agents:<project fingerprint of name>`; URL from `GPU_QUEUE_URL` (env or `~/.gpu/credentials` via `CredentialResolver`); missing URL → ConfigError naming the variable.

- [ ] **Step 1: Write the failing tests**: `class TestRedisBackend(QueueBackendContract)` against fakeredis (clock = patched server `TIME`), and the same class against a real Redis when `GPU_TEST_REDIS_URL` is set (skipped otherwise); `test_lease_expiry_uses_the_server_clock` (client `time.time` patched 1 h ahead → claim still valid until the server clock passes `lease_until`); `test_missing_url_names_the_variable`; `test_importing_gpu4agents_without_redis_works` (subprocess with `redis` blocked via `sys.modules["redis"] = None`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_backend_redis.py -v` → FAIL.
- [ ] **Step 3: Implement** with Lua scripts for claim, renew, update, settle (atomic check of token + lease against `redis.call('TIME')`); a sorted set of QUEUED ids by submit time; a hash per job; a stream `audit` for control actions. `now()` reads `TIME`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `"Redis queue backend (optional extra queue-redis), atomic by server-side scripts and the server's clock, passing the backend contract."`

### Task 6: Output stores and verified delivery

**Files:**
- Create: `gpu4agents/queue/outputs.py`
- Modify: `pyproject.toml` (extra `queue-s3 = ["boto3>=1.34"]`; dev gains `moto[s3]>=5`)
- Test: `tests/test_queue_outputs.py`

**Interfaces:**
- Produces:
```python
@dataclass class OutputRef: location: str; files: list[dict]   # each {"path", "bytes", "sha256"}
class OutputStore(ABC):
    def put_bundle(self, job_id: str, path: Path) -> str
    def get_bundle(self, ref: str, dest: Path) -> Path
    def put_outputs(self, job_id: str, attempt: int, src_dir: Path) -> OutputRef
    def verify(self, ref: OutputRef) -> list[str]                # problems; [] means intact
    def fetch_previous(self, ref: OutputRef, dest: Path) -> None # for $JAMBU_PREVIOUS_OUTPUT_DIR
class LocalOutputStore(OutputStore): __init__(root: Path)        # root = <project>/gpu-outputs
class S3OutputStore(OutputStore): __init__(url: str)             # GPU_OUTPUTS_URL = s3://bucket/prefix, endpoint via AWS_ENDPOINT_URL
def load_output_store(spec: str, config: RuntimeConfig) -> OutputStore
```
  Local layout: bundles in `<state_dir>/queue/bundles/<job id>.tar.gz`, outputs in `gpu-outputs/<job id>/` (the latest successful attempt; failed attempts in `gpu-outputs/<job id>/.attempt-<n>/`), plus `MANIFEST.json`.

- [ ] **Step 1: Write the failing tests**: `test_local_put_and_verify_round_trip`, `test_a_truncated_upload_fails_verification` (truncate one stored file → `verify` lists it), `test_manifest_has_sha256_per_file`, `test_s3_store_round_trip` (moto), `test_bundle_round_trip`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_outputs.py -v` → FAIL.
- [ ] **Step 3: Implement**; S3 writes the manifest last, so a manifest present means every file was written.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `"Output stores for queue jobs (local, S3-compatible) with a sha256 manifest checked before a job counts as delivered."`

### Task 7: Preflight

**Files:**
- Create: `gpu4agents/queue/preflight.py`
- Test: `tests/test_queue_preflight.py`

**Interfaces:**
- Produces:
```python
@dataclass class Limits: timeout_s: float; max_cost_usd: Optional[float]; deadline_margin_s: float
@dataclass class PreflightContext: hourly_usd: float; card_vram_gb: Optional[float]; defaults: QueueDefaults; attempt: int
@dataclass class PreflightResult: ok: bool; limits: Optional[Limits]; reason: str = ""
def default_preflight(job: QueueJob, ctx: PreflightContext) -> PreflightResult
def load_preflight(spec: Optional[str]) -> Callable[[QueueJob, PreflightContext], PreflightResult]
```
  Rules of the default: timeout = manifest `timeout` or `defaults.timeout`; max cost = manifest or defaults; reject when `min_vram_gb > card_vram_gb` (reason `"needs N GB, the pool's card has M GB"`), and when `max_cost_usd < hourly_usd * timeout_s / 3600` (reason states both numbers). A custom preflight that raises or returns a non-`PreflightResult` → rejected with the exception text (the coordinator never crashes on it).

- [ ] **Step 1: Write the failing tests**: `test_job_values_win_over_defaults`, `test_defaults_fill_missing_values`, `test_rejects_a_job_bigger_than_the_card`, `test_rejects_a_cost_limit_its_timeout_cannot_meet`, `test_a_custom_preflight_is_loaded_by_name`, `test_a_crashing_custom_preflight_rejects_the_job`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_preflight.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `"Per-job preflight: limits from the job and the queue defaults, refusals that explain why, and a hook for a project's own rules."`

### Task 8: Scaling decision (pure)

**Files:**
- Create: `gpu4agents/queue/scaler.py`
- Test: `tests/test_queue_scaler.py`

**Interfaces:**
- Produces:
```python
@dataclass class MachineView: slot: str; state: str; hourly_usd: float; running_jobs: int
@dataclass class ScaleInput: now: float; queued: int; oldest_queued_age_s: Optional[float]; machines: list[MachineView]; pool_usd_per_hour: float; next_offer_usd: Optional[float]; min_machines: int; max_machines: int; max_usd_per_hour: Optional[float]; scale_up_after_s: float; jobs_per_machine: int; last_open_at: Optional[float]; open_backoff_s: float
@dataclass class ScaleDecision: open: int; keepalive: list[str]; reason: str; blocked_by: Optional[str]   # "max_machines" | "max_usd_per_hour" | "no_offer" | "backoff" | None
def decide(inp: ScaleInput) -> ScaleDecision
```
  Rules: machines in states `provisioning`, `checking`, `ready`, `busy` count; free capacity = sum of `jobs_per_machine - running_jobs` over ready/busy plus `jobs_per_machine` per provisioning/checking. Open when `queued > free capacity` and `oldest_queued_age_s >= scale_up_after_s`, or when the count is below `min_machines`; open `min(needed, max_machines - count, affordable)` where affordable counts how many `next_offer_usd` fit under `max_usd_per_hour - pool_usd_per_hour`. Never closes a machine (idle_timeout does). `keepalive` = the `min_machines` oldest ready machines with no jobs.

- [ ] **Step 1: Write the failing tests**: `test_opens_when_jobs_wait_past_the_threshold`, `test_waits_before_the_threshold`, `test_counts_machines_still_booting_as_capacity`, `test_stops_at_max_machines` (`blocked_by == "max_machines"`), `test_stops_at_the_spend_cap` (`blocked_by == "max_usd_per_hour"`), `test_keeps_min_machines_open_with_an_empty_queue`, `test_respects_open_backoff`, `test_no_offer_blocks_without_failing`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_scaler.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `"Pure scaling decision for the pool: open when jobs wait, within max_machines and the hourly spend cap, never close a working machine."`

### Task 9: Pool spend accounting

**Files:**
- Create: `gpu4agents/queue/spend.py`
- Test: `tests/test_queue_spend.py`

**Interfaces:**
- Produces: `def pool_usd_per_hour(machines: list[PoolMachine]) -> float` (running at `hourly_usd`; stopped at `stopped_hourly_usd`; held-for-outputs at `hourly_usd`); `class SpendLedger: __init__(path: Path)`, `record(slot: str, from_ts: float, to_ts: float, usd_per_hour: float) -> float`, `total() -> float`, `by_job() -> dict[str, float]` (job cost = machine rate × job run time ÷ jobs sharing the machine). `PoolMachine` is defined in Task 10; this task defines its fields used here in `gpu4agents/queue/pool_types.py`: `slot, instance_id, state, hourly_usd, stopped_hourly_usd, ready_at, deadline_at, jobs: list[str], health_failures, last_seen_at, opened_at`.

- [ ] **Step 1: Write the failing tests**: `test_stopped_machines_count_their_disk`, `test_held_machines_count_full_rate`, `test_job_cost_splits_a_shared_machine`, `test_ledger_survives_restart` (new `SpendLedger` on the same path keeps `total()`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_spend.py -v` → FAIL.
- [ ] **Step 3: Implement** (`pool_types.py` + `spend.py`; ledger is JSONL append).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `"Pool spend: what running, stopped and held machines cost per hour, and a ledger that attributes cost to each job."`

### Task 10: The pool

**Files:**
- Create: `gpu4agents/queue/pool.py`
- Test: `tests/test_queue_pool.py` (FakeProvider + the in-process watchdog runner from `tests/test_jobs.py`, moved to `tests/conftest.py` as a shared fixture)

**Interfaces:**
- Consumes: `Engine(..., state_dir=, slot=)`, `Engine.set_deadline`, `JobClient` (`core/job.py`), `Observer`, `PoolMachine`, `pool_usd_per_hour`.
- Produces:
```python
@dataclass class HealthResult: ok: bool; detail: str
class Pool:
    def __init__(self, config: RuntimeConfig, observer: Observer, provider: Optional[GPUProvider] = None, clock=time.time)
    def machines(self) -> list[PoolMachine]
    def open(self) -> PoolMachine                      # returns at once (state "provisioning"); setup runs in a thread
    def poll(self) -> list[PoolMachine]                # reconcile each; transitions to ready/unhealthy/unresponsive/gone
    def health_check(self, m: PoolMachine) -> HealthResult
    def extend_deadline(self, m: PoolMachine, until: float) -> None
    def keepalive(self, m: PoolMachine) -> None        # touches activity
    def dispatch(self, m: PoolMachine, remote_id: str, bundle: Path, manifest: dict, env: dict) -> str   # "started" | "already_running"
    def job_info(self, m: PoolMachine, remote_id: str) -> dict
    def fetch_outputs(self, m: PoolMachine, remote_id: str, dest: Path) -> list[str]
    def cancel(self, m: PoolMachine, remote_id: str) -> None
    def retire(self, m: PoolMachine, reason: str) -> None   # pull what can be pulled, destroy, mark host bad
    def next_offer_usd(self) -> Optional[float]
    def card_vram_gb(self) -> Optional[float]
```
  Slots are `s1, s2, …` (reused after `gone`); each lives in `<state_dir>/pool/<slot>/`. State machine: provisioning → checking (first health check) → ready ↔ busy → (unhealthy | unresponsive | gone | retired). `unresponsive` = watchdog unreachable for `unresponsive_after` while the provider says running. A `dispatch` that gets 409 from the watchdog returns `"already_running"`. The built-in health check is a job `python3 -c "import torch; assert torch.cuda.is_available(); torch.ones(1).cuda().sum().item()"` (or `nvidia-smi -L` when torch is absent), plus `min_free_vram_gb` when set; a project command replaces it. Open failures (capacity, provisioning) back off 30 s doubling to 5 min, emitting `pool_cap_blocked`/`provider_error` once per change.

- [ ] **Step 1: Write the failing tests**: `test_open_returns_at_once_and_becomes_ready_after_the_health_check`, `test_a_failed_health_check_retires_and_marks_the_host_bad`, `test_unresponsive_after_the_grace_period` (watchdog stopped; provider running; clock advanced 301 s → state `unresponsive`, event `pool_machine_unresponsive`), `test_a_vanished_machine_is_gone`, `test_dispatch_twice_returns_already_running`, `test_extend_deadline_reaches_the_watchdog`, `test_no_capacity_backs_off_and_keeps_jobs_queued` (FakeProvider raises CapacityUnavailableError; three `open` calls within 30 s produce one provider call), `test_slots_have_their_own_state_and_labels`, `test_cold_start_is_measured` (`gpu_pool_cold_start_seconds_count == 1` after ready).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_pool.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** `"A pool of machines from one gpu.yaml: open without blocking, health-checked before work, unresponsive and vanished machines detected, deadlines pushed to each watchdog."`

### Task 11: The coordinator

**Files:**
- Create: `gpu4agents/queue/coordinator.py`
- Create: `tests/queue_fakes.py` (`FakePool` implementing the Pool interface in memory with scripted job outcomes)
- Test: `tests/test_queue_coordinator.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
```python
@dataclass class TickReport: claimed: int; dispatched: int; settled: int; requeued: int; opened: int; violations: int
class Coordinator:
    def __init__(self, config: RuntimeConfig, backend: QueueBackend, outputs: OutputStore, pool: Pool, observer: Observer, preflight=None, worker_id: Optional[str] = None, clock=time.time)
    def tick(self) -> TickReport
    def run(self, stop: threading.Event) -> None          # tick every queue.tick; standby while another coordinator holds the lease
```
  One tick, in order: (1) renew the coordinator lease, or try to acquire it (standby otherwise; on acquire, `recover()`); (2) `pool.poll()`; (3) renew the lease of every job in hand; (4) for each running attempt: finished → deliver (fetch → `put_outputs` → `verify` → settle SUCCEEDED/FAILED, or keep DELIVERING and retry next tick on any delivery problem); non-zero exit and `health_check` fails → requeue with `UNHEALTHY` and retire the machine; machine `gone`/`unresponsive` → try `fetch_outputs`, retire, requeue with `MACHINE_LOST`/`UNRESPONSIVE`; `cancel_requested` → `pool.cancel`, settle CANCELLED; (5) while a ready machine has a free slot and the queue is not paused: claim → preflight (reject → settle FAILED with `PREFLIGHT_REJECTED`) → `extend_deadline(now + limits.timeout_s + limits.deadline_margin_s)` (never earlier than the current one) → dispatch with remote id `f"{job.id}-a{n}"` and, for n > 1, `JAMBU_PREVIOUS_OUTPUT_DIR` filled from the last attempt's outputs; (6) `scaler.decide` → `pool.open()` × n, `keepalive`; (7) light audit (Task 12) every 12 ticks; (8) metrics `write()`. A job with `len(attempts) >= max_attempts` is settled FAILED (`queue_job_dead`) instead of requeued. `LeaseLost` on any write → drop the job from memory, emit `queue_stale_write_refused`, never retry the write.
  `recover()`: for every non-final job not held, look for its last attempt's remote id on the pool machines; running there → adopt it (re-claim when the lease expired, keep monitoring); finished there → deliver; nowhere → requeue with `MACHINE_LOST`.

- [ ] **Step 1: Write the failing tests** (FakePool, LocalBackend, LocalOutputStore, fake clock):
  `test_a_job_runs_and_its_outputs_are_delivered_and_verified`,
  `test_jobs_run_back_to_back_on_one_warm_machine` (3 jobs, 1 machine → 1 `open`, 3 dispatches),
  `test_the_deadline_moves_with_each_job`,
  `test_a_lost_machine_requeues_the_job_on_another`,
  `test_a_crash_and_failed_health_check_count_as_the_machine`,
  `test_a_plain_failure_is_the_job_s_and_is_not_retried`,
  `test_max_attempts_ends_in_failed_with_every_reason`,
  `test_retry_gets_the_previous_outputs` (env of attempt 2 has `JAMBU_PREVIOUS_OUTPUT_DIR` containing attempt 1's files),
  `test_preflight_rejection_settles_failed_with_the_reason`,
  `test_crash_after_dispatch_adopts_the_running_job_instead_of_starting_it_twice` (dispatch, drop the Coordinator object, new Coordinator after lease expiry → FakePool sees one start, job ends SUCCEEDED),
  `test_delivery_failure_keeps_the_job_delivering_and_retries`,
  `test_a_standby_coordinator_takes_over_after_the_lease_expires`,
  `test_pause_stops_claims_but_not_running_jobs`,
  `test_cancel_reaches_the_running_job`,
  `test_a_stale_lease_write_is_dropped_not_retried`,
  `test_every_decision_is_an_event` (after one full job: events contain `queue_job_claimed`, `queue_job_started`, `queue_outputs_stored`, `queue_job_succeeded`, `pool_scale_decision`, `pool_deadline_extended`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_coordinator.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** `"The queue coordinator: claims within free capacity, runs each job through preflight, deadline, dispatch and verified delivery, retries on machine faults, and recovers in-flight jobs after a crash."`

### Task 12: Invariant auditor

**Files:**
- Create: `gpu4agents/queue/audit.py`
- Test: `tests/test_queue_audit.py`

**Interfaces:**
- Produces: `@dataclass class Violation: code: str; job_id: Optional[str]; slot: Optional[str]; detail: str; repaired: bool`; `def audit(backend, pool, outputs, observer, *, repair: bool) -> list[Violation]`.
  Codes and repairs:
  - `I1_running_without_machine` (repair: requeue `MACHINE_LOST`)
  - `I2_duplicate_execution`: a remote job running whose queue job is not RUNNING on that slot (repair: cancel the remote job)
  - `I3_outputs_unverified` (repair: job back to DELIVERING)
  - `I4_orphan_machine`: a provider instance with this project's label prefix and fingerprint and no slot (repair: retire it)
  - `I5_spend_over_cap` (no repair, alert)
  - `I6_job_past_deadline` (repair: cancel and requeue `TIMEOUT`)
  - `I7_stale_coordinator`: lease holder heartbeat > 3 ticks (no repair, alert)

  Each violation emits `queue_invariant_violated` and increments `gpu_queue_invariant_violations_total{code}`.

- [ ] **Step 1: Write the failing tests**: one per code, each building the broken state with FakePool + LocalBackend and asserting the code, the repair and the metric; `test_a_clean_pool_has_no_violations`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_audit.py -v` → FAIL.
- [ ] **Step 3: Implement**; the coordinator calls `audit(..., repair=True)` every 12 ticks and on `recover()`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** `"An auditor for the queue's guarantees (nothing run twice, nothing lost, nothing unverified, no orphan, no spend over the cap), repairing what it safely can."`

### Task 13: CLI, Python API, MCP tools and controls

**Files:**
- Create: `gpu4agents/cli/queue.py`, `gpu4agents/queue/api.py`
- Modify: `gpu4agents/cli/main.py` (`app.add_typer(queue.queue_app, name="queue")`; `"queue"` added to `QUIET_COMMANDS` only for `run`), `gpu4agents/agent/mcp.py` (tools below), `gpu4agents/cli/timer.py` (`render_queue_unit(...)` for a `gpu queue run` systemd unit/LaunchAgent, same escaping rules)
- Test: `tests/test_queue_cli.py`, `tests/test_agent.py` (MCP list)

**Interfaces:**
- Produces (`api.py`): `submit(directory: Path | str = ".", *, key: Optional[str] = None, project: Optional[Path] = None, env: Optional[dict] = None) -> str`; `status(job_id: str, project=None) -> dict`; `wait(job_id: str, timeout_s: Optional[float] = None, project=None) -> dict`; `cancel(job_id, project=None) -> dict`; `outputs(job_id, dest: Optional[Path] = None, project=None) -> Path`. `submitted_by` = `GPU_QUEUE_SUBMITTER` or `user@host`.
- Produces (CLI, all with `--json`):
  - `gpu queue submit [DIR] [--key K] [--env K=V]...` → `{"job_id"}`
  - `gpu queue status [JOB]` (no JOB: queue summary with stats, pool, spend, coordinator)
  - `gpu queue ls [--state S]`
  - `gpu queue cancel JOB`
  - `gpu queue retry JOB`
  - `gpu queue pull JOB [--to DIR]`
  - `gpu queue run [--once] [--metrics-port P]` (P serves `/metrics` on 127.0.0.1 only)
  - `gpu queue pause`, `gpu queue resume`
  - `gpu queue drain [--now]` (pause + let machines idle out; `--now` also cancels and requeues running jobs and retires every machine)
  - `gpu queue audit [--repair]`
  - `gpu queue stats` (Observer snapshot)
  - `gpu queue install-service [--uninstall]`

  Every control action writes `queue_control` (action, actor) to the backend's audit trail and the event log.
- Produces (MCP): `gpu_queue_submit`, `gpu_queue_status`, `gpu_queue_cancel`, `gpu_queue_pull`, `gpu_queue_control` (`action` in pause/resume/drain; `drain` with `now=true` is destructive and needs `confirm=true`). `gpu_queue_status` is read-only annotated.

- [ ] **Step 1: Write the failing tests**: `test_submit_returns_an_id_at_once_and_queues_the_bundle`, `test_submit_rejects_bad_jobs_and_stores_nothing` (no `run` / bundle over 2 GB via patched size / bad `timeout` → exit code non-zero, `backend.list()` empty; paused → queued with `paused: true` in output), `test_submit_is_idempotent_with_a_key`, `test_status_summary_has_pool_spend_and_coordinator`, `test_control_actions_are_audited_with_the_actor`, `test_drain_now_requires_confirm_in_mcp`, `test_python_api_submit_and_wait` (coordinator `run --once` loop driven in a thread with FakePool), `test_queue_service_unit_runs_gpu_queue_run`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_queue_cli.py tests/test_agent.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** `"gpu queue: submit, status, cancel, retry, pull, run, pause, resume, drain, audit and stats from the CLI, a Python API and MCP tools, every control action audited."`

### Task 14: Sweep and timer know pool machines

**Files:**
- Modify: `gpu4agents/core/sweep.py` (guard every `<state_dir>/pool/*/` as its own project state; pull outputs before destroy as today)
- Modify: `gpu4agents/cli/_common.py` (`warn_stopped_billing` includes pool slots)
- Test: `tests/test_sweep.py`

- [ ] **Step 1: Write the failing tests**: `test_sweep_guards_pool_slots_of_a_registered_project` (a stopped pool machine past `stopped_ttl` with the coordinator gone → destroyed by `sweep --apply`), `test_sweep_pulls_a_pool_job_s_outputs_before_destroying`, `test_stopped_billing_warning_counts_pool_machines`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_sweep.py -k pool -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** `"gpu sweep and the stopped-billing warning cover pool machines, so a dead coordinator still ends in a destroy with outputs saved."`

### Task 15: Docs, acceptance guarantees, version 0.14.0

**Files:**
- Create: `QUEUE.md` (copied to `site/web/queue.md`), `docs/QUEUE_BACKENDS.md` (writing a backend or an output store; running `QueueBackendContract`)
- Modify: `docs/ACCEPTANCE.md` (guarantees 24–31: one per anti-goal, each naming its test), `docs/CONFIG.md` (`queue:` table), `docs/USAGE.md` (`gpu queue …`), `docs/LIFECYCLE.md` (deadline rule), `docs/INTEGRATION.md` (Python API), both `SKILL.md` copies (when to use queue mode; never start `gpu queue run` without the user's yes and caps), `CHANGELOG.md` (0.14.0), `README.md` only the queue paragraph, `site/web/docs/index.html` (a "Queue mode" section with the observation table and alert thresholds; MCP tools; event vocabulary), `site/web/llms.txt` (queue.md link), version files.
- Test: `tests/test_agent.py` (skill copies identical; version check string), `tests/test_docs.py` (new: every event constant in `core/events.py` appears in the docs page; every `gpu queue` subcommand appears in USAGE.md)

- [ ] **Step 1: Write the failing tests** (`tests/test_docs.py` as above; version 0.14.0 in plugin manifests).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_docs.py tests/test_agent.py -v` → FAIL.
- [ ] **Step 3: Write the docs.** QUEUE.md alert thresholds:
  - coordinator heartbeat age > 3 ticks
  - any invariant violation
  - oldest pending delivery > 10 min
  - retry rate > 20% over 1 h
  - pool spend ≥ 90% of `max_usd_per_hour`
  - oldest queued job > 2 × `scale_up_after` + cold start p50 while `blocked_by` is set

  Do not touch the user's uncommitted hero rewrite in `site/web/index.html`; only `docs/index.html`, `queue.md` and `llms.txt` lines for the queue.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** `"Queue mode docs (QUEUE.md, backends guide, config, usage, lifecycle, acceptance guarantees, site docs) and version 0.14.0."`

### Task 16: Real run on Vast.ai (needs the user's yes before renting)

**Files:**
- Create: `docs/handoff/2026-11-xx-queue-mode-field-run.md` (results)

- [ ] **Step 1:** Show the user the plan:
  - 20 jobs of 1–3 min each (`examples/queue-smoke/`: a CUDA matmul that writes a JSON result);
  - `max_machines: 2` and `max_usd_per_hour` at 2× the cheapest 16 GB offer from `gpu offers`;
  - the estimated total cost.

  Wait for an explicit yes.
- [ ] **Step 2:** Run `INFISICAL_PATH=/jambu-gpu devrun gpu queue run` in one terminal and submit the 20 jobs. Mid-run, destroy one machine with `gpu destroy` from its slot directory. Then kill the coordinator, restart it, and record each of these:
  - every job ends `succeeded`;
  - `gpu queue audit` reports none;
  - `gpu ls` is empty after `idle_timeout`.
- [ ] **Step 3:** Run a second time with the coordinator killed and not restarted. Record that the machines end at their deadline or idle timeout and that `gpu ls` is empty.
- [ ] **Step 4:** Write the results (events, metrics snapshot, cost) to the handoff file and commit it.
