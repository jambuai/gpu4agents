# Writing a queue backend or an output store

Queue mode ([QUEUE.md](../QUEUE.md)) holds jobs in a queue and files in an output store. Both are contracts, so a company broker, a database you already run or another bucket can stand in for the built-ins (`local` SQLite and `redis`; `local` directory and `s3`). A backend is supported when it passes the same suite the built-ins pass.

Name yours in `gpu.yaml` as `module:factory`. The factory receives the project's `RuntimeConfig` and returns the object:

```yaml
gpu_runtime:
  queue:
    backend: acme_queue:create     # def create(config) -> QueueBackend
    outputs: acme_store:create     # def create(config) -> OutputStore
```

Addresses and keys do not go in `gpu.yaml`. Read them from the environment or through `gpu4agents.core.credentials.CredentialResolver`, and never print them.

## The queue backend

Subclass `gpu4agents.queue.backend.QueueBackend` (`queue/backend.py` has the signatures and the docstrings that are the specification). The shape:

| Group | Methods | The rule |
| --- | --- | --- |
| Time | `now()` | Every lease and age uses this clock, never the caller's. Hosts with skewed clocks share one queue |
| Submit | `submit(job)` | Idempotent on `job.key`: a known key returns the existing job. A new job is stamped `submitted_at = now()`. Job ids are at most 60 characters |
| Claim | `claim(worker, lease_s, job_id=None)` | Atomic: oldest `QUEUED` job, invisible to other claims until its lease expires or it is settled. Returns nothing while paused. With `job_id`, claims exactly that job if it is queued, even while paused: a coordinator taking back a run already on a machine is not new work |
| Lease | `renew`, `update`, `requeue`, `settle` | A stale or expired token raises `LeaseLost` and writes nothing. `settle` is final. An expired claim goes back to `QUEUED`, or to `CANCELLED` when a cancel was requested |
| Control | `request_cancel`, `retry`, `set_paused`, `paused`, `request_drain`, `drain_requested`, `clear_drain` | Cancel of a queued job is immediate. `retry` takes a `FAILED` job back to `QUEUED`, keeping attempts. Each action is recorded with its actor, append-only |
| Read | `get`, `list(states, limit)`, `stats()` | `stats` gives counts by state and the age of the oldest queued job, which the scaler needs |
| Coordinator | `acquire_coordinator`, `renew_coordinator`, `release_coordinator`, `coordinator()` | One lease per queue. A second coordinator waits and takes over when it expires |

The job payload stays small: the bundle (the packed job directory) lives in the output store, and the job holds a reference (`bundle_ref`).

Redis notes for the built-in: Redis 5 or newer, a single node (replicas for failover are fine, Cluster is not: the atomic steps are Lua scripts over several keys). Replication in Redis is asynchronous, so a failover can lose a write the old primary had already acknowledged, such as a claim, a lease renewal or the record of a dispatch. The effect is a duplicate claim: the job is claimed again with the same attempt number, so its remote id is the same. If that dispatch reaches the machine that already has the run, the machine answers 409 and the run is adopted, not started twice. If the run never started there, the fence refuses it before the job goes back to the queue. If the second dispatch goes to another machine, both copies run until the auditor's next pass (I2, every 12 ticks) cancels the copy the queue does not place there. Without replicas or failover none of this happens. If you use them, keep failovers rare and expect a `queue_stale_write_refused` event or an I2 violation after one. The keyspace is `gpu4agents:<namespace>`, where the namespace is `queue.namespace`, else `GPU_QUEUE_NAMESPACE`, else a name from the config fingerprint (which changes when `gpu.yaml`'s runtime changes: set `queue.namespace`).

### Run the contract suite

`gpu4agents.queue.testing.QueueBackendContract` is the conformance suite. Subclass it in a test module and provide two fixtures: `backend`, and `clock`, an object whose `advance(seconds)` moves `backend.now()`. A backend therefore needs an injectable clock or offset for tests (the Redis backend adds an offset to the server's `TIME`).

```python
import pytest
from gpu4agents.queue.testing import QueueBackendContract, make_job

class FakeClock:
    def __init__(self): self.t = 1000.0
    def now(self): return self.t
    def advance(self, s): self.t += s

@pytest.fixture
def clock(): return FakeClock()

@pytest.fixture
def backend(clock): return AcmeBackend(clock=clock.now)

class TestAcmeBackend(QueueBackendContract):
    pass
```

It covers no double claim, lease expiry, stale tokens refused, idempotent submit, final states, cancel and retry, claiming a specific job. Both built-ins subclass it (`tests/test_queue_backend_local.py`, `tests/test_queue_backend_redis.py`).

## The output store

Subclass `gpu4agents.queue.outputs.OutputStore`:

| Method | What it does |
| --- | --- |
| `put_bundle(job_id, path) -> ref` | Store the packed job; return the reference the queue keeps |
| `get_bundle(ref, dest)` | Put it back for the machine |
| `delete_bundle(ref)` | Remove it (after a failed submit, or when the job is final). Must be safe to call twice |
| `put_outputs(job_id, attempt, src_dir, success=True) -> OutputRef` | Store a directory and a sha256 manifest of its files (`path`, `bytes`, `sha256`). Re-putting an attempt leaves no stale keys, and a failed put leaves no manifest |
| `verify(ref) -> list[str]` | Problems found (missing, extra, wrong size or hash); empty means verified. A job is marked `succeeded` only after this is empty |
| `fetch_previous(ref, dest)` | Copy an earlier attempt's verified files into `dest`, for `$JAMBU_PREVIOUS_OUTPUT_DIR` |

Paths that escape the store (`..`, absolute) are refused. A store that is unreachable raises: the coordinator then keeps the fetched files under `<state_dir>/queue/work/`, retries, and never retires a machine for it.

`tests/test_queue_outputs.py` shows the checks the built-in stores pass (round trip, truncated upload fails verification, re-put, escaping paths).
