"""The queue from Python: submit a job directory, follow it, cancel it, get its outputs.

This is what ``gpu queue ...`` and the MCP tools stand on. Everything reads
``gpu.yaml`` (``gpu_runtime.queue`` must be present) and talks to the configured queue
backend and output store; nothing here rents a machine, the coordinator does
(``gpu queue run``).

    from gpu4agents.queue import api
    job_id = api.submit("train/", key="train-2026-11-12")   # idempotent on the key
    info = api.wait(job_id, timeout_s=3600)
    path = api.outputs(job_id)
"""

from __future__ import annotations

import getpass
import os
import secrets
import socket
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

from ..core import events as ev
from ..core import job as jobmod
from ..core.config import RuntimeConfig, find_config, load_config
from ..core.errors import ConfigError, StateError
from ..core.events import EventLog
from .backend import QueueBackend, load_backend
from .models import FINAL_STATES, JobState, QueueJob
from .observe import Observer
from .outputs import OutputRef, OutputStore, load_output_store
from .pool_types import MachineState
from .spend import machine_usd_per_hour, pool_usd_per_hour

SUBMITTER_ENV = "GPU_QUEUE_SUBMITTER"
# Remote ids are "<id>-a<n>" under the watchdog's 64-character cap.
JOB_ID_MAX = 60
WAIT_POLL_S = 1.0


def submitter() -> str:
    """Who is acting: ``GPU_QUEUE_SUBMITTER`` or ``user@host``."""
    named = os.environ.get(SUBMITTER_ENV, "").strip()
    if named:
        return named
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no passwd entry in a container
        user = "unknown"
    return f"{user}@{socket.gethostname()}"


def new_job_id() -> str:
    return "q-" + secrets.token_hex(6)


# ---------------------------------------------------------------------------
# what a caller needs open
# ---------------------------------------------------------------------------


@dataclass
class Runtime:
    config: RuntimeConfig
    backend: QueueBackend
    outputs: OutputStore
    observer: Observer


def load_queue_config(project: Optional[Path | str] = None) -> RuntimeConfig:
    """The gpu.yaml of ``project`` (a directory, searched upwards, or the file itself)."""
    path = Path(project).expanduser() if project is not None else None
    if path is not None and path.is_dir():
        found = find_config(path)
        if found is None:
            raise ConfigError(f"no gpu.yaml in {path} or any parent")
        path = found
    config = load_config(path)
    if config.queue is None:
        raise ConfigError("queue mode is not on: add a `queue:` block under gpu_runtime in gpu.yaml")
    return config


def make_observer(config: RuntimeConfig) -> Observer:
    return Observer(EventLog(config.state_dir / "logs"), config.state_dir / "queue" / "metrics.prom")


def open_runtime(project: Optional[Path | str] = None) -> Runtime:
    config = load_queue_config(project)
    queue = config.queue
    return Runtime(
        config=config,
        backend=load_backend(queue.backend, config),
        outputs=load_output_store(queue.outputs, config),
        observer=make_observer(config),
    )


def build_pool(config: RuntimeConfig, observer: Observer) -> Any:
    """The machine pool (a seam: tests replace it)."""
    from .pool import Pool

    return Pool(config, observer)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def _manifest_for(directory: Path, env: Optional[dict]) -> dict:
    if not directory.is_dir():
        raise ConfigError(f"{directory} is not a directory")
    manifest = jobmod.load_manifest(directory)
    if not (manifest.run or "").strip():
        raise ConfigError(f"{jobmod.MANIFEST_NAME} in {directory} needs a `run:` command: that is what the machine runs")
    data = manifest.model_dump(exclude_none=True)
    merged = {**manifest.env, **{str(k): str(v) for k, v in (env or {}).items()}}
    if merged:
        data["env"] = merged
    return data


def submit_job(
    directory: Path | str = ".",
    *,
    key: Optional[str] = None,
    project: Optional[Path] = None,
    env: Optional[dict] = None,
) -> dict:
    """Queue a job directory. Returns ``{"job_id", "key", "state", "paused", "existing"}``.

    Nothing is stored unless the job is valid. A submission to a paused queue is queued
    and reported with ``paused: true``. The same ``key`` always gives back the same job.
    """
    directory = Path(directory).expanduser()
    if key is not None and not (1 <= len(key) <= 200):
        raise ConfigError("--key must be 1 to 200 characters")
    manifest = _manifest_for(directory, env)
    rt = open_runtime(project)
    who = submitter()

    if key is not None:  # a known key stores nothing: the job is already there
        for known in rt.backend.list(limit=100_000):
            if known.key == key:
                return _submitted(rt, known, existing=True)

    job_id = new_job_id()
    with tempfile.TemporaryDirectory(prefix="gpu-queue-bundle-") as tmp:
        target = Path(tmp) / f"{job_id}.tar.gz"
        built = jobmod.build_bundle(directory, manifest.get("exclude") or [], target)
        ref = rt.outputs.put_bundle(job_id, target)
    try:
        job = rt.backend.submit(
            QueueJob(
                id=job_id, key=key or job_id, bundle_ref=ref, manifest=manifest,
                submitted_at=0.0, submitted_by=who,
            )
        )
    except BaseException:
        _drop_bundle(rt, ref)  # nothing was queued: the bundle must not stay behind
        raise
    existing = job.id != job_id
    if existing:
        _drop_bundle(rt, ref)  # another submit with this key won the race
    else:
        rt.observer.emit(
            ev.QUEUE_JOB_SUBMITTED, job_id=job.id, key=job.key, submitted_by=who,
            files=built["files"], bytes=built["bytes"],
        )
    return _submitted(rt, job, existing=existing)


def _drop_bundle(rt: Runtime, ref: str) -> None:
    try:
        rt.outputs.delete_bundle(ref)
    except Exception:  # noqa: BLE001 - an orphan bundle is a leak, not a reason to fail the call
        pass


def _submitted(rt: Runtime, job: QueueJob, existing: bool) -> dict:
    return {
        "job_id": job.id, "key": job.key, "state": job.state.value,
        "paused": rt.backend.paused(), "existing": existing,
    }


def submit(
    directory: Path | str = ".",
    *,
    key: Optional[str] = None,
    project: Optional[Path] = None,
    env: Optional[dict] = None,
) -> str:
    """Queue a job directory and return its id at once."""
    return submit_job(directory, key=key, project=project, env=env)["job_id"]


def _get(rt: Runtime, job_id: str) -> QueueJob:
    job = rt.backend.get(job_id)
    if job is None:
        raise StateError(f"no job {job_id!r} in this queue")
    return job


def describe(job: QueueJob) -> dict:
    """The job as a dict, safe to print: env names stay, their values are hidden."""
    data = job.to_dict()
    manifest = dict(data.get("manifest") or {})
    if isinstance(manifest.get("env"), dict):
        manifest["env"] = {k: "***" for k in manifest["env"]}
    data["manifest"] = manifest
    data["final"] = job.state in FINAL_STATES
    return data


def status(job_id: str, project=None) -> dict:
    return describe(_get(open_runtime(project), job_id))


def wait(job_id: str, timeout_s: Optional[float] = None, project=None) -> dict:
    """Block until the job is SUCCEEDED, FAILED or CANCELLED and return its record.
    After ``timeout_s`` the current record comes back with ``timed_out: true``."""
    rt = open_runtime(project)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        job = _get(rt, job_id)
        if job.state in FINAL_STATES:
            return describe(job)
        if deadline is not None and time.monotonic() >= deadline:
            return {**describe(job), "timed_out": True}
        time.sleep(WAIT_POLL_S if deadline is None else max(0.0, min(WAIT_POLL_S, deadline - time.monotonic())))


def listing(state: Optional[str] = None, project=None, limit: int = 100) -> list[dict]:
    rt = open_runtime(project)
    states = None
    if state:
        try:
            states = {JobState(state.lower())}
        except ValueError as exc:
            raise ConfigError(f"unknown state {state!r}; use one of {', '.join(s.value for s in JobState)}") from exc
    return [
        {
            "id": j.id, "state": j.state.value, "key": j.key, "submitted_by": j.submitted_by,
            "submitted_at": j.submitted_at, "attempts": len(j.attempts), "machine": j.machine,
            "cancel_requested": j.cancel_requested,
        }
        for j in rt.backend.list(states, limit=limit)
    ]


def _audited(rt: Runtime, action: str, fn: Callable[[str], Any], **fields: Any) -> Any:
    """Run ``fn(actor)``, then record ``queue_control`` in the event log: ``ok`` true after it
    succeeded, ``ok`` false with the error type when it raised (the backend audits its own calls)."""
    who = submitter()
    try:
        result = fn(who)
    except Exception as exc:
        rt.observer.emit(ev.QUEUE_CONTROL, action=action, actor=who, ok=False, error=type(exc).__name__, **fields)
        raise
    extra = result if isinstance(result, dict) and action.startswith("drain") else {}
    rt.observer.emit(
        ev.QUEUE_CONTROL, action=action, actor=who, ok=True,
        **{k: v for k, v in extra.items() if k in ("drain", "performed_by", "requeued", "remaining")}, **fields,
    )
    return result


def cancel(job_id: str, project=None) -> dict:
    rt = open_runtime(project)
    _get(rt, job_id)
    return describe(_audited(rt, "cancel", lambda who: rt.backend.request_cancel(job_id, who), job_id=job_id))


def retry(job_id: str, project=None) -> dict:
    rt = open_runtime(project)
    _get(rt, job_id)

    def do(who: str) -> QueueJob:
        try:
            return rt.backend.retry(job_id, who)
        except ValueError as exc:
            raise StateError(str(exc)) from exc

    return describe(_audited(rt, "retry", do, job_id=job_id))


def outputs(job_id: str, dest: Optional[Path] = None, project=None) -> Path:
    """Put a finished job's verified outputs in ``dest`` (default ``gpu-outputs/<job id>``) and return it."""
    rt = open_runtime(project)
    job = _get(rt, job_id)
    if not job.outputs_ref:
        raise StateError(
            f"job {job_id} has no outputs yet (it is {job.state.value}); "
            "`gpu queue status` shows where it is"
        )
    ref = OutputRef(**job.outputs_ref)
    target = Path(dest).expanduser() if dest else rt.config.project_dir / jobmod.OUTPUTS_DIRNAME / job_id
    try:
        same = Path(ref.location).resolve() == target.resolve()
    except OSError:
        same = False
    problems = rt.outputs.verify(ref) if same else []
    if same:
        if problems:
            raise StateError("stored outputs failed verification: " + "; ".join(problems))
        return target
    try:
        rt.outputs.fetch_previous(ref, target)
    except ValueError as exc:
        raise StateError(str(exc)) from exc
    return target


# ---------------------------------------------------------------------------
# the queue as a whole
# ---------------------------------------------------------------------------


def _pool_view(rt: Runtime) -> dict:
    try:
        pool = build_pool(rt.config, rt.observer)
        machines = pool.machines()
    except Exception as exc:  # noqa: BLE001 - no provider credentials here: report, do not fail
        return {"machines": None, "usd_per_hour": None, "error": type(exc).__name__}
    live = [m for m in machines if m.state not in (MachineState.GONE, MachineState.RETIRED)]
    spend = getattr(pool, "spend", None)
    return {
        "machines": [
            {
                "slot": m.slot, "state": m.state, "hourly_usd": m.hourly_usd,
                "billing_usd_per_hour": machine_usd_per_hour(m), "instance_id": m.instance_id,
                "jobs": list(m.jobs), "deadline_at": m.deadline_at,
            }
            for m in live
        ],
        "usd_per_hour": round(pool_usd_per_hour(machines), 6),
        "spent_usd": round(spend.total(), 6) if spend is not None else None,
    }


def summary(project=None) -> dict:
    """Stats, pool machines (state, hourly), pool spend per hour, the coordinator lease holder, paused."""
    rt = open_runtime(project)
    stats = rt.backend.stats()
    lease = rt.backend.coordinator()
    now = rt.backend.now()
    return {
        "queue": {"by_state": stats.by_state, "oldest_queued_age_s": stats.oldest_queued_age_s},
        "paused": rt.backend.paused(),
        "drain_requested": rt.backend.drain_requested(),
        "coordinator": (
            {"owner": lease["owner"], "lease_remaining_s": round(max(0.0, lease["lease_until"] - now), 3)}
            if lease else None
        ),
        "pool": _pool_view(rt),
        "backend": getattr(rt.backend, "name", None),
    }


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        name, _, value = line.rpartition(" ")
        if not name or name.startswith("#"):
            continue
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def stats(project=None) -> dict:
    """The coordinator's metrics (its last write of ``metrics.prom``) and the queue's own numbers."""
    rt = open_runtime(project)
    path = rt.observer.metrics_path
    metrics: dict[str, float] = {}
    age = None
    try:
        metrics = parse_metrics(path.read_text())
        age = round(max(0.0, time.time() - path.stat().st_mtime), 3)
    except OSError:
        pass
    q = rt.backend.stats()
    return {
        "queue": {"by_state": q.by_state, "oldest_queued_age_s": q.oldest_queued_age_s},
        "metrics": metrics,
        "metrics_age_s": age,
        "metrics_file": str(path),
    }


def pause(project=None) -> dict:
    rt = open_runtime(project)

    def do(who: str) -> dict:
        rt.backend.set_paused(True, who)
        return {"paused": True, "actor": who}

    return _audited(rt, "pause", do)


def resume(project=None) -> dict:
    """Start jobs again. A drain request still waiting for its coordinator is dropped: it
    would pause the queue again."""
    rt = open_runtime(project)

    def do(who: str) -> dict:
        cleared = rt.backend.drain_requested()
        if cleared:
            rt.backend.clear_drain()
        rt.backend.set_paused(False, who)
        return {"paused": False, "actor": who, "drain_cleared": cleared}

    return _audited(rt, "resume", do)


def drain(project=None, now: bool = False) -> dict:
    """Pause the queue and let the machines idle out. With ``now``: also send running jobs
    back to the queue (not failed) and retire every machine, pulling their outputs.

    ``now`` acts through the queue: the request is recorded in the backend and the running
    coordinator performs it on its next tick. With no coordinator holding the lease it is
    done here, directly; what cannot be done yet (a machine that will not retire, a job
    still leased to a vanished coordinator) stays requested for the next coordinator."""
    rt = open_runtime(project)

    def do(who: str) -> dict:
        rt.backend.set_paused(True, who)
        if not now:
            return {"paused": True, "drain": "idle", "actor": who}
        rt.backend.request_drain(who)
        if rt.backend.coordinator() is not None:
            return {"paused": True, "drain": "requested", "performed_by": "coordinator", "actor": who}
        from .coordinator import Coordinator

        coordinator = Coordinator(rt.config, rt.backend, rt.outputs, build_pool(rt.config, rt.observer), rt.observer)
        try:
            result = coordinator.drain_now(who)
        finally:
            coordinator.release()
        if not result.get("performed"):
            return {"paused": True, "drain": "requested", "performed_by": "coordinator", "actor": who}
        done = not result["remaining"]
        if done:
            rt.backend.clear_drain()
        return {"paused": True, "drain": "done" if done else "partial", "actor": who, **result}

    return _audited(rt, "drain_now" if now else "drain", do)


def audit_queue(project=None, repair: bool = False) -> dict:
    """Check the queue's guarantees once; with ``repair`` fix what is safe."""
    from .audit import audit

    rt = open_runtime(project)

    def do(who: str) -> dict:
        pool = build_pool(rt.config, rt.observer)
        found = audit(rt.backend, pool, rt.outputs, rt.observer, repair=repair, config=rt.config)
        return {"violations": [asdict(v) for v in found], "repair": repair, "ok": not found}

    return _audited(rt, "audit", do, repair=repair)


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def serve_metrics(observer: Observer, port: int) -> ThreadingHTTPServer:
    """Serve ``/metrics`` (Prometheus text) on 127.0.0.1 only, from a daemon thread.
    ``port`` 0 picks a free one (``server.server_address``). Stop with ``server.shutdown()``."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path.split("?", 1)[0] != "/metrics":
                self.send_error(404)
                return
            body = observer.render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:  # silence the access log
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    except OSError as exc:
        raise ConfigError(f"cannot serve metrics on 127.0.0.1:{port}: {exc.strerror or exc}") from exc
    threading.Thread(target=server.serve_forever, name="queue-metrics", daemon=True).start()
    return server


__all__ = [
    "submit", "status", "wait", "cancel", "outputs", "retry", "listing", "summary", "stats", "pause",
    "resume", "drain", "audit_queue", "submitter", "serve_metrics",
]
