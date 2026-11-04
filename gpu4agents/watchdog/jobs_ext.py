"""Job runner: an extension the CLI uploads to the watchdog (``POST /extend``).

It runs inside the watchdog process, in that module's globals (``_LOCK``,
``_STATE``, ``log_event``, ``save_state``, ...), and uses the stdlib only. A
job is a directory of code plus a shell command. The CLI uploads the directory
as a tar.gz; the job runs in its own process group with stdout and stderr in
one log; whatever it writes to ``$JAMBU_OUTPUT_DIR`` goes back as a tar.gz.

While a job runs it is a workload, and the watchdog heartbeats it itself: the
machine stays up with no laptop connected, and the idle clock starts when the
job ends. max_lifetime and the session budget still apply.

Routes (behind X-Jambu-Token, like every other control route):
    GET  /jobs                    -> {jobs: [records]}
    PUT  /jobs/<id>/bundle        <- tar.gz of the code (optional)
    POST /jobs/<id>/start         -> {command, setup, env, timeout_s, label}
    GET  /jobs/<id>               -> the record, plus the output files so far
    GET  /jobs/<id>/log?offset=N  -> {data, offset, size, done}
    GET  /jobs/<id>/outputs       -> tar.gz of the output directory
    POST /jobs/<id>/cancel

On disk, under JOBS_DIR/<id>/: bundle.tar.gz, src/ (the code, and the job's
working directory), out/ (outputs), job.log, job.json (the record), and rc
(the exit code, written by the job's own shell, so a respawned watchdog still
learns how a job ended).
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
        _atomic_write,
        log_event,
        save_state,
        touch,
    )

import json
import os
import re
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

# Kept across a re-upload of this extension: the processes are still ours.
_JOB_PROCS = globals().get("_JOB_PROCS") or {}
_JOB_LOCK = globals().get("_JOB_LOCK") or threading.RLock()
# A new version of this code retires the monitor threads of the old one.
_JOB_GENERATION = int(globals().get("_JOB_GENERATION") or 0) + 1


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
        os.remove(path + ".part")
        return 400, {"error": "upload cut short (%d bytes missing)" % remaining}
    os.replace(path + ".part", path)
    touch("job_upload")
    return 200, {"ok": True, "bytes": length}


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
        timeout = body.get("timeout_s")
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
        }
        job_write(record)
        _JOB_PROCS[job_id] = proc
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
    log_event("job_started", job_id=job_id, label=record["label"])
    threading.Thread(target=job_monitor, args=(job_id, _JOB_GENERATION), daemon=True).start()
    return 200, {"ok": True, "job": record}


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
    request.send_response(200)
    request.send_header("Content-Type", "application/gzip")
    request.end_headers()
    # HTTP/1.0: the body ends when the connection closes, so it can stream.
    with tarfile.open(fileobj=request.wfile, mode="w|gz") as tar:
        tar.add(out, arcname=".")


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
        request._reply(200, {"jobs": job_list()})
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
EXT_ROUTES["jobs"] = jobs_route
