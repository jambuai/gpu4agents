"""Job runner: an extension the CLI uploads to the watchdog (``POST /extend``).

It runs inside the watchdog process, in that module's globals (``_LOCK``,
``_STATE``, ``log_event``, ``save_state``, ...), and uses the stdlib only. A
job is a directory of code plus a shell command. The CLI uploads the directory
as a tar.gz; the job runs in its own process group with stdout and stderr in
one log; whatever it writes to ``$JAMBU_OUTPUT_DIR`` goes back as a tar.gz.

While a job runs it is a workload, and the watchdog heartbeats it itself: the
machine stays up with no laptop connected, and the idle clock starts when the
job ends. max_lifetime and the session budget still apply.

Results are never thrown away unseen. A job's outputs count as saved once
``GET /jobs/<id>/outputs`` has streamed them after the job ended (and nothing
changed since). While any are unsaved, a shutdown the policy picks becomes a
stop where the disk survives one; where it would not, an idle machine is held
up to OUTPUTS_HOLD_S for the CLI to fetch them (see ``outputs_guard``).

A *service* job (``"service": true``) is a model server instead: it is not a
workload, so it never holds the machine up by itself running; the requests it
serves do, through the activity probe on its /metrics. Its pid becomes the
runtime pid (a server that dies while loading ends the startup wait) and its
log becomes the runtime log (`gpu logs`).

Routes (behind X-Jambu-Token, like every other control route):
    GET  /jobs                    -> {jobs: [records], uploads: {id: {state, age_s}},
                                      jobs_dir_age_s}
    PUT  /jobs/<id>/bundle        <- tar.gz of the code (optional)
    POST /jobs/<id>/start         -> {command, setup, env, timeout_s, label,
                                      service, digest}
    GET  /jobs/<id>               -> the record, plus the output files so far
    GET  /jobs/<id>/log?offset=N  -> {data, offset, size, done}
    GET  /jobs/<id>/outputs       -> tar.gz of the output directory
    POST /jobs/<id>/cancel
    POST /jobs/<id>/refuse        -> {refused: true} or, when it started, {refused:
                                     false, job: record}

On disk, under JOBS_DIR/<id>/: bundle.tar.gz, src/ (the code, and the job's
working directory), out/ (outputs), job.log, job.json (the record), and rc
(the exit code, written by the job's own shell, so a respawned watchdog still
learns how a job ended).

``GET /jobs`` also names every job that has code on the machine but was never
started (``uploads``: "uploading" while its bundle is still arriving, "uploaded"
once it is complete, "empty" for a directory with neither; ``age_s`` is how long
ago it last changed), and how old the jobs directory is (``jobs_dir_age_s``, from
a marker written when the runner first finds it). A queue coordinator that takes
over uses both to tell a run that never started from one still on its way: a
directory older than the dispatch that holds nothing for the job means the job
never reached this machine. Before it sends such a job back to the queue it fences
it with ``POST /jobs/<id>/refuse``: the runner records the id as "refused" and
deletes its code, so a late upload or start from the coordinator that died gets a
409 and never runs. A job that did start is returned instead, to be adopted.
"""

# Loaded with exec() into the watchdog's globals. Imported directly (tests,
# linting), it borrows the same names from the agent body.
if "log_event" not in globals():  # pragma: no cover - only in standalone import
    from gpu4agents.watchdog.agent_body import (  # noqa: F401
        _LOCK,
        _STATE,
        _CONFIG,
        _STOPPED,
        EXT_ROUTES,
        TERMINATE_HOOKS,
        _atomic_write,
        log_event,
        save_state,
        touch,
    )
    from gpu4agents.core.policy import ACTION_DESTROY, ACTION_STOP, REASON_IDLE  # noqa: F401

import json
import os
import re
import shutil
import signal
import subprocess
import tarfile
import threading
import time
from urllib.parse import parse_qs

JOBS_DIR = os.environ.get("JAMBU_JOBS_DIR", "/workspace/jobs")
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
JOB_HEARTBEAT_S = float(os.environ.get("JAMBU_JOB_HEARTBEAT_S", "10"))
JOB_CANCEL_GRACE_S = float(os.environ.get("JAMBU_JOB_CANCEL_GRACE_S", "10"))
JOB_BUNDLE_MAX_BYTES = 2 * 1024**3
JOB_LOG_CHUNK_MAX = 4 * 1024**2
JOB_OUTPUT_LIST_MAX = 200
# Where a stop would wipe the disk too (no persistent disk, or no stop at all),
# results nobody downloaded hold an idle machine up this long after the last
# job ended, for `gpu sweep` or `gpu job pull` to fetch them.
OUTPUTS_HOLD_S = float(os.environ.get("JAMBU_OUTPUTS_HOLD_S", "7200"))

# Kept across a re-upload of this extension: the processes are still ours.
_JOB_PROCS = globals().get("_JOB_PROCS") or {}
_JOB_LOCK = globals().get("_JOB_LOCK") or threading.RLock()
# A new version of this code retires the monitor threads of the old one.
_JOB_GENERATION = int(globals().get("_JOB_GENERATION") or 0) + 1
# When the runner first found this jobs directory (it goes with the directory).
JOBS_DIR_MARKER = ".jambu-jobs-since"


def _job_path(job_id, *parts):
    return os.path.join(JOBS_DIR, job_id, *parts)


def job_read(job_id):
    try:
        with open(_job_path(job_id, "job.json")) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def job_write(record):
    _atomic_write(_job_path(record["id"], "job.json"), json.dumps(record, indent=2))


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    try:
        with open("/proc/%d/stat" % pid) as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return True


def _read_rc(job_id):
    try:
        with open(_job_path(job_id, "rc")) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _extract(bundle, dest):
    """Unpack the uploaded code, refusing anything that would land outside ``dest``."""
    with tarfile.open(bundle, "r:*") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
            return
        base = os.path.realpath(dest)
        for member in tar.getmembers():
            target = os.path.realpath(os.path.join(dest, member.name))
            inside = target == base or target.startswith(base + os.sep)
            if not inside or member.issym() or member.islnk() or member.isdev():
                raise ValueError("unsafe path in bundle: %s" % member.name)
        tar.extractall(dest)


def _outputs(job_id):
    out = _job_path(job_id, "out")
    files, total = [], 0
    for root, _dirs, names in os.walk(out):
        for name in sorted(names):
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            total += size
            if len(files) < JOB_OUTPUT_LIST_MAX:
                files.append({"path": os.path.relpath(path, out), "bytes": size})
    return files, total


def _outputs_signature(job_id):
    """(files, bytes, newest mtime) of the output directory; None when empty."""
    count, total, newest = 0, 0, 0.0
    for root, _dirs, names in os.walk(_job_path(job_id, "out")):
        for name in names:
            try:
                info = os.stat(os.path.join(root, name))
            except OSError:
                continue
            count += 1
            total += info.st_size
            newest = max(newest, info.st_mtime)
    return [count, total, round(newest, 3)] if count else None


def outputs_unsaved(record):
    """True when this job wrote results nobody has downloaded since it ended."""
    if record.get("service"):
        return False
    signature = _outputs_signature(record["id"])
    if signature is None:
        return False
    if record.get("state") == "running":
        return True
    return record.get("outputs_saved") != signature


def outputs_guard(choice, cfg):
    """Turn a shutdown that would lose unsaved job results into one that keeps them."""
    action, reason, original = choice
    if _STATE.get("pending_shutdown") or reason == "stop_failed_escalation":
        return choice  # asked for explicitly, or a stop that cannot happen
    unsaved = [r for r in job_list() if outputs_unsaved(r)]
    if not unsaved:
        return choice
    ids = [r["id"] for r in unsaved]
    if cfg.stop_supported and _CONFIG.get("stop_keeps_disk"):
        if action == ACTION_DESTROY:
            _log_once("outputs_kept", ids, reason=reason, action="stop")
            return ACTION_STOP, reason, original
        return choice
    # A stop or destroy would take the results with it.
    if reason == REASON_IDLE:
        ended = max(float(r.get("finished_at") or time.time()) for r in unsaved)
        if time.time() - ended < OUTPUTS_HOLD_S:
            _log_once("outputs_hold", ids, hold_seconds=OUTPUTS_HOLD_S)
            return None
    _log_once("outputs_lost", ids, reason=reason, action=action)
    return choice


def _log_once(name, ids, **detail):
    key = name + ":" + ",".join(ids)
    with _LOCK:
        if _STATE.get("outputs_logged") == key:
            return
        _STATE["outputs_logged"] = key
        save_state()
    log_event(name, job_ids=ids, **detail)


def job_upload(request, job_id):
    if job_read(job_id) is not None:
        return 409, {"error": "job %s already started" % job_id}
    try:
        length = int(request.headers.get("Content-Length") or 0)
    except ValueError:
        length = 0
    if length <= 0:
        return 411, {"error": "Content-Length required"}
    if length > JOB_BUNDLE_MAX_BYTES:
        return 413, {"error": "bundle over %d bytes" % JOB_BUNDLE_MAX_BYTES}
    os.makedirs(_job_path(job_id), exist_ok=True)
    path = _job_path(job_id, "bundle.tar.gz")
    remaining = length
    with open(path + ".part", "wb") as handle:
        while remaining > 0:
            chunk = request.rfile.read(min(1 << 20, remaining))
            if not chunk:
                break
            handle.write(chunk)
            remaining -= len(chunk)
    if remaining:
        _remove(path + ".part")
        return 400, {"error": "upload cut short (%d bytes missing)" % remaining}
    with _JOB_LOCK:
        if job_read(job_id) is not None:
            # Refused (or started) while the code was on its way: it never runs from here.
            _remove(path + ".part")
            return 409, {"error": "job %s already started" % job_id}
        try:
            os.replace(path + ".part", path)
        except OSError as exc:
            _remove(path + ".part")
            return 500, {"error": "could not store the bundle of job %s: %s" % (job_id, exc)}
    touch("job_upload")
    return 200, {"ok": True, "bytes": length}


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def job_refuse(job_id):
    """Fence a job that never started: from now on its upload and start get a 409.
    A job that did start is returned as it is (the caller adopts it)."""
    with _JOB_LOCK:
        record = job_read(job_id)
        if record is not None and record.get("state") != "refused":
            return 200, {"ok": True, "refused": False, "job": record}
        if record is None:
            os.makedirs(_job_path(job_id), exist_ok=True)
            record = {"id": job_id, "state": "refused", "refused_at": time.time(), "started_at": None}
            job_write(record)
        for name in ("bundle.tar.gz", "bundle.tar.gz.part"):
            _remove(_job_path(job_id, name))
        shutil.rmtree(_job_path(job_id, "src"), ignore_errors=True)
    log_event("job_refused", job_id=job_id)
    return 200, {"ok": True, "refused": True}


def job_start(job_id, body):
    command = str(body.get("command") or "").strip()
    if not command:
        return 400, {"error": "command is required"}
    with _JOB_LOCK:
        if job_read(job_id) is not None:
            return 409, {"error": "job %s already started" % job_id}
        src, out = _job_path(job_id, "src"), _job_path(job_id, "out")
        os.makedirs(src, exist_ok=True)
        os.makedirs(out, exist_ok=True)
        bundle = _job_path(job_id, "bundle.tar.gz")
        if os.path.exists(bundle):
            try:
                _extract(bundle, src)
            except (tarfile.TarError, ValueError, OSError) as exc:
                return 400, {"error": "cannot unpack the bundle: %s" % exc}
        setup = str(body.get("setup") or "").strip()
        with open(_job_path(job_id, "command.sh"), "w") as handle:
            handle.write(command + "\n")
        if setup:
            with open(_job_path(job_id, "setup.sh"), "w") as handle:
                handle.write(setup + "\n")

        env = dict(os.environ)
        # The provider credential the watchdog stops the machine with is not the job's.
        for key in _CONFIG.get("env") or {}:
            env.pop(str(key), None)
        env.setdefault("HF_HOME", "/workspace/hf")
        env.update(
            {
                "JAMBU_JOB_ID": job_id,
                "JAMBU_JOB_DIR": _job_path(job_id),
                "JAMBU_OUTPUT_DIR": out,
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.update({str(k): str(v) for k, v in (body.get("env") or {}).items()})
        # The shell writes the exit code itself: it outlives a watchdog respawn.
        script = (
            'cd "$JAMBU_JOB_DIR/src" && '
            '{ [ ! -f ../setup.sh ] || bash ../setup.sh; } && '
            "bash ../command.sh; "
            'rc=$?; echo "$rc" > ../rc; exit "$rc"'
        )
        with open(_job_path(job_id, "job.log"), "ab") as log:
            proc = subprocess.Popen(
                ["bash", "-c", script],
                cwd=src,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        now = time.time()
        service = bool(body.get("service"))
        timeout = None if service else body.get("timeout_s")
        record = {
            "id": job_id,
            "label": str(body.get("label") or ""),
            "command": command,
            "setup": setup or None,
            "state": "running",
            "pid": proc.pid,
            "started_at": now,
            "finished_at": None,
            "returncode": None,
            "timeout_s": float(timeout) if timeout else None,
            "session_started_at": _STATE.get("session_started_at"),
            "service": service,
            "digest": str(body.get("digest") or "") or None,
        }
        job_write(record)
        _JOB_PROCS[job_id] = proc
    if service:
        _service_publish(job_id, proc.pid)
        touch("service_start")
    else:
        with _LOCK:
            _STATE.setdefault("workloads", {})[job_id] = {
                "started_at": now,
                "last_heartbeat": now,
                "label": "job " + (record["label"] or job_id),
                "state": "running",
            }
            _STATE["last_activity"] = now
            _STATE["starting"] = False
            save_state()
    log_event("job_started", job_id=job_id, label=record["label"], service=service)
    threading.Thread(target=job_monitor, args=(job_id, _JOB_GENERATION), daemon=True).start()
    return 200, {"ok": True, "job": record}


def _service_publish(job_id, pid):
    """Make the service the runtime: its pid for liveness, its log for /logs."""
    pid_path = _CONFIG.get("runtime_pid_path")
    log_path = (_CONFIG.get("logs") or {}).get("runtime")
    try:
        if pid_path:
            os.makedirs(os.path.dirname(pid_path), exist_ok=True)
            _atomic_write(pid_path, str(pid))
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            if os.path.lexists(log_path):
                os.remove(log_path)
            os.symlink(_job_path(job_id, "job.log"), log_path)
    except OSError as exc:
        log_event("service_publish_error", job_id=job_id, error=str(exc))


def _job_done(job_id, record):
    """(finished, returncode) for a running job."""
    proc = _JOB_PROCS.get(job_id)
    if proc is not None:
        code = proc.poll()
        if code is None:
            return False, None
        rc = _read_rc(job_id)
        return True, rc if rc is not None else code
    pid = record.get("pid")
    if pid and _pid_alive(pid):
        return False, None
    return True, _read_rc(job_id)


def _job_finish(record, rc):
    now = time.time()
    with _JOB_LOCK:
        record = job_read(record["id"]) or record
        if record.get("state") != "running":
            return
        reason = record.get("cancel_reason")
        if reason:
            state = reason
        elif rc is None:
            state = "lost"
        else:
            state = "succeeded" if rc == 0 else "failed"
        record.update(state=state, returncode=rc, finished_at=now)
        job_write(record)
        _JOB_PROCS.pop(record["id"], None)
    with _LOCK:
        workload = (_STATE.get("workloads") or {}).get(record["id"])
        if workload is not None:
            workload["state"] = "finished" if state == "succeeded" else "failed"
            workload["finished_at"] = now
            workload["returncode"] = rc
        _STATE["last_activity"] = now
        save_state()
    log_event(
        "job_finished",
        job_id=record["id"],
        state=state,
        returncode=rc,
        seconds=round(now - float(record.get("started_at") or now), 1),
    )


def job_monitor(job_id, generation):
    while not _STOPPED.is_set() and generation == _JOB_GENERATION:
        record = job_read(job_id)
        if record is None or record.get("state") != "running":
            return
        done, rc = _job_done(job_id, record)
        if done:
            _job_finish(record, rc)
            return
        timeout = record.get("timeout_s")
        now = time.time()
        if timeout and now - float(record["started_at"]) > timeout and not record.get("cancel_reason"):
            job_cancel(job_id, "timeout")
        if record.get("service"):
            # Serving is not working: only the requests it answers are activity.
            _STOPPED.wait(JOB_HEARTBEAT_S)
            continue
        with _LOCK:
            workload = (_STATE.get("workloads") or {}).get(job_id)
            if workload is not None:
                workload["last_heartbeat"] = now
            _STATE["last_activity"] = now
            save_state()
        _STOPPED.wait(JOB_HEARTBEAT_S)


def job_cancel(job_id, reason="cancelled"):
    with _JOB_LOCK:
        record = job_read(job_id)
        if record is None:
            return 404, {"error": "no job %s" % job_id}
        if record.get("state") != "running":
            return 200, {"ok": True, "state": record.get("state")}
        record["cancel_reason"] = reason
        record["cancel_requested_at"] = time.time()
        job_write(record)
    pid = int(record["pid"])
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    log_event("job_cancel_requested", job_id=job_id, reason=reason)

    def _kill_later():
        time.sleep(JOB_CANCEL_GRACE_S)
        current = job_read(job_id)
        if current and current.get("state") == "running":
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass

    threading.Thread(target=_kill_later, daemon=True).start()
    return 202, {"ok": True, "state": "cancelling"}


def job_info(job_id):
    record = job_read(job_id)
    if record is None:
        return 404, {"error": "no job %s" % job_id}
    files, total = _outputs(job_id)
    end = record.get("finished_at") or time.time()
    try:
        log_bytes = os.path.getsize(_job_path(job_id, "job.log"))
    except OSError:
        log_bytes = 0
    record.update(
        outputs_unsaved=outputs_unsaved(record),
        outputs=files,
        outputs_count_capped=len(files) >= JOB_OUTPUT_LIST_MAX,
        outputs_bytes=total,
        log_bytes=log_bytes,
        elapsed_s=round(end - float(record.get("started_at") or end), 1),
    )
    return 200, record


def job_log(job_id, query):
    record = job_read(job_id)
    if record is None:
        return 404, {"error": "no job %s" % job_id}
    try:
        offset = max(0, int((query.get("offset") or [0])[0]))
        limit = min(JOB_LOG_CHUNK_MAX, max(1, int((query.get("limit") or [JOB_LOG_CHUNK_MAX])[0])))
    except ValueError:
        return 400, {"error": "offset and limit must be integers"}
    path = _job_path(job_id, "job.log")
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(limit)
    except OSError:
        size, data = 0, b""
    if len(data) == limit and b"\n" in data:
        # Stop on a line boundary so a multi-byte character is never split.
        data = data[: data.rindex(b"\n") + 1]
    return 200, {
        "data": data.decode("utf-8", "replace"),
        "offset": offset + len(data),
        "size": size,
        "done": record.get("state") != "running",
        "state": record.get("state"),
    }


def job_outputs(request, job_id):
    out = _job_path(job_id, "out")
    if not os.path.isdir(out):
        request._reply(404, {"error": "no outputs for job %s" % job_id})
        return
    touch("job_outputs")
    record = job_read(job_id) or {}
    finished = record.get("state") not in (None, "running")
    signature = _outputs_signature(job_id) if finished else None
    request.send_response(200)
    request.send_header("Content-Type", "application/gzip")
    request.end_headers()
    # HTTP/1.0: the body ends when the connection closes, so it can stream.
    with tarfile.open(fileobj=request.wfile, mode="w|gz") as tar:
        tar.add(out, arcname=".")
    request.wfile.flush()
    if finished:
        # Fully sent: these results now exist off the machine.
        with _JOB_LOCK:
            current = job_read(job_id) or record
            current["outputs_saved"] = signature
            job_write(current)
        log_event("job_outputs_saved", job_id=job_id, files=signature[0] if signature else 0)


def jobs_dir_age():
    """Seconds since the runner first found the jobs directory (0 for a new one)."""
    path = os.path.join(JOBS_DIR, JOBS_DIR_MARKER)
    try:
        with open(path) as handle:
            since = float(handle.read().strip())
    except (OSError, ValueError):
        since = time.time()
        try:
            os.makedirs(JOBS_DIR, exist_ok=True)
            _atomic_write(path, repr(since))
        except OSError:
            pass
    return max(0.0, time.time() - since)


def job_uploads():
    """Jobs with a directory but no record: code on its way or waiting for a start."""
    try:
        names = sorted(os.listdir(JOBS_DIR))
    except OSError:
        return {}
    now, found = time.time(), {}
    for name in names:
        if not JOB_ID_PATTERN.match(name) or os.path.exists(_job_path(name, "job.json")):
            continue
        for state, path in (
            ("uploading", _job_path(name, "bundle.tar.gz.part")),
            ("uploaded", _job_path(name, "bundle.tar.gz")),
            ("empty", _job_path(name)),
        ):
            try:
                changed = os.stat(path).st_mtime
            except OSError:
                continue
            found[name] = {"state": state, "age_s": round(max(0.0, now - changed), 3)}
            break
    return found


def job_list():
    try:
        names = sorted(os.listdir(JOBS_DIR))
    except OSError:
        return []
    records = [r for r in (job_read(n) for n in names) if r]
    return sorted(records, key=lambda r: r.get("started_at") or 0)


def jobs_route(request, method, path, body):
    parts = path.strip("/").split("/")
    if parts[0] != "jobs":
        return False
    if len(parts) == 1:
        if method != "GET":
            return False
        # Uploads first: a job that starts in between then shows up as a record.
        uploads = job_uploads()
        records = job_list()
        for record in records:
            record["outputs_unsaved"] = outputs_unsaved(record)
        request._reply(200, {"jobs": records, "uploads": uploads, "jobs_dir_age_s": jobs_dir_age()})
        return True
    job_id = parts[1]
    if not JOB_ID_PATTERN.match(job_id) or len(parts) > 3:
        request._reply(400, {"error": "bad job path"})
        return True
    action = parts[2] if len(parts) == 3 else ""
    if (method, action) == ("PUT", "bundle"):
        request._reply(*job_upload(request, job_id))
    elif (method, action) == ("POST", "start"):
        request._reply(*job_start(job_id, body or {}))
    elif (method, action) == ("POST", "cancel"):
        request._reply(*job_cancel(job_id))
    elif (method, action) == ("POST", "refuse"):
        request._reply(*job_refuse(job_id))
    elif (method, action) == ("GET", ""):
        request._reply(*job_info(job_id))
    elif (method, action) == ("GET", "log"):
        request._reply(*job_log(job_id, parse_qs(request.path.partition("?")[2])))
    elif (method, action) == ("GET", "outputs"):
        job_outputs(request, job_id)
    else:
        request._reply(404, {"error": "not found"})
    return True


def jobs_resume():
    """Watch the jobs that were running before this code was (re)loaded."""
    session = _STATE.get("session_started_at")
    for record in job_list():
        if record.get("state") != "running":
            continue
        job_id = record["id"]
        if record.get("session_started_at") != session and job_id not in _JOB_PROCS:
            # The machine restarted under the job: its process is gone.
            _job_finish(record, _read_rc(job_id))
            continue
        threading.Thread(target=job_monitor, args=(job_id, _JOB_GENERATION), daemon=True).start()


jobs_resume()
jobs_dir_age()  # the marker, from the first time this runner is loaded
EXT_ROUTES["jobs"] = jobs_route
TERMINATE_HOOKS["jobs"] = outputs_guard
