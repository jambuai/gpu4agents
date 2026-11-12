"""Remote lifecycle watchdog (spec section 14).

This file is NOT imported at runtime by the CLI. Its source is concatenated
with ``core/policy.py`` and the provider-supplied action source into a single
stdlib-only script that runs *on the rented instance*, so lifecycle
enforcement survives the local CLI crashing, the laptop sleeping or the
network dying.

Contract with the CLI (all routes require ``X-Jambu-Token``):
    GET  /health              -> liveness
    GET  /state               -> snapshot + current policy decision
    GET  /events?limit=N      -> structured events recorded on the instance
    GET  /logs?name=X&lines=N -> tail of a configured log (runtime|onstart|watchdog)
    POST /activity            -> generic activity mark
    POST /ready               -> model server finished starting
    POST /workload/start      -> {id, label}
    POST /workload/heartbeat  -> {id}
    POST /workload/end        -> {id, returncode, state}
    POST /lock                -> {ttl_seconds|until|null}
    POST /unlock
    POST /shutdown            -> {action: stop|destroy, reason}
    POST /extend              -> {name, source}: load more routes (see below)

Extensions: the boot script rides in a size-capped provider field, so features
the CLI needs only later (the job runner) are uploaded once the machine is up.
``/extend`` saves the source under EXT_DIR and runs it in this module's
globals; it registers a route handler in EXT_ROUTES. Saved extensions are
loaded again when the supervisor respawns the agent.

Sessions: the boot script exports ``JAMBU_SESSION_STARTED_AT`` once per
container start. A value different from the stored one means the instance was
restarted, so the persisted "terminated" flag and stale workloads are reset and
the guard starts over. The same value means a supervisor respawn: keep state.
"""

# The generated script inlines core/policy.py above this body. When this file
# is imported directly (tests, linting) we pull the same names from the package.
if "evaluate" not in globals():  # pragma: no cover - only in standalone import
    from gpu4agents.core.policy import (  # noqa: F401
        ACTION_DESTROY,
        ACTION_NONE,
        ACTION_STOP,
        Decision,
        PolicyConfig,
        Snapshot,
        Workload,
        derive_state,
        evaluate,
    )

import base64
import hashlib
import hmac
import json
import math
import os
import sys
import threading
import time
import traceback
import urllib.request
import zlib
from urllib.parse import parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONFIG_PATH = os.environ.get("JAMBU_WATCHDOG_CONFIG", "/etc/jambu/watchdog.json")
STATE_PATH = os.environ.get("JAMBU_WATCHDOG_STATE", "/var/lib/jambu/activity.json")
EVENTS_PATH = os.environ.get("JAMBU_WATCHDOG_EVENTS", "/var/log/jambu/events.jsonl")
EXT_DIR = os.environ.get("JAMBU_WATCHDOG_EXT", "/opt/jambu/ext")

_LOCK = threading.RLock()
_STATE = {}
_CONFIG = {}
_STOPPED = threading.Event()
# Serializes provider calls; the guard loop and /shutdown may race.
_ACTION_LOCK = threading.Lock()
_PROBE = {}
# name -> handler(request, method, path, body) returning True when it answered.
EXT_ROUTES = {}
# name -> f(choice, cfg): may change or veto (None) a shutdown (see jobs_ext).
TERMINATE_HOOKS = {}
# name -> digest of the loaded source, so the CLI uploads only what changed.
_EXT = {}

# Seconds of continuously failing stops before giving up on the disk and
# destroying (config key escalate_after_s). Long enough to ride out a provider
# API outage: losing the disk to a 3-minute blip is worse than 30 more minutes.
ESCALATE_AFTER_S = 1800.0
PROBE_ERROR_LOG_EVERY_S = 600.0
LOG_LINES_MAX = 2000


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def _atomic_write(path, payload):
    directory = os.path.dirname(path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        pass
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def log_event(event, **fields):
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "source": "watchdog",
        "provider": _CONFIG.get("provider"),
        "instance_id": (_STATE or {}).get("instance_id") or _CONFIG.get("instance_id"),
    }
    record.update(fields)
    line = json.dumps(record, default=str)
    try:
        os.makedirs(os.path.dirname(EVENTS_PATH) or ".", exist_ok=True)
        with open(EVENTS_PATH, "a") as handle:
            handle.write(line + "\n")
    except OSError:
        pass
    print("[jambu-watchdog] " + line, flush=True)
    return record


def _session_from_env():
    raw = os.environ.get("JAMBU_SESSION_STARTED_AT")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_state():
    now = time.time()
    default = {
        "created_at": float(_CONFIG.get("created_at") or now),
        "last_activity": now,
        "starting": True,
        "locked": False,
        "workloads": {},
        "terminated": False,
        "session_started_at": None,
    }
    stored = None
    try:
        with open(STATE_PATH) as handle:
            stored = json.load(handle)
        default.update(stored)
    except (OSError, ValueError):
        stored = None

    session = _session_from_env()
    if stored is None:
        default["session_started_at"] = session if session is not None else now
        log_event("session_started", session_started_at=default["session_started_at"], first_boot=True)
        return default
    if session is None:
        # Old boot scripts and tests: no session marker, behave as before.
        if default.get("session_started_at") is None:
            default["session_started_at"] = default["created_at"]
        return default
    if default.get("session_started_at") == session:
        return default  # supervisor respawn inside the same container start
    return _new_session(default, session, now)


def _new_session(state, session, now):
    """The container was started again (e.g. `gpu` restarted a stopped instance).

    Container disk survives a stop, so the previous session's "terminated" flag
    would otherwise disarm the guard on the restarted, billing machine.
    """
    abandoned = []
    for key, record in (state.get("workloads") or {}).items():
        if record.get("state") in ("running", "queued"):
            record["state"] = "failed"
            record["finished_at"] = now
            record["failure"] = "session_restarted"
            abandoned.append(key)
    previous = state.get("session_started_at")
    state.update(
        {
            "session_started_at": session,
            "terminated": False,
            "terminated_at": None,
            "starting": True,
            "last_activity": now,
            "stop_failures": 0,
            "stop_failing_since": None,
            "action_failures": 0,
            "pending_shutdown": None,
        }
    )
    log_event(
        "session_started",
        session_started_at=session,
        previous_session_started_at=previous,
        abandoned_workloads=abandoned,
    )
    return state


def save_state():
    try:
        _atomic_write(STATE_PATH, json.dumps(_STATE, indent=2, default=str))
    except OSError:
        pass


def touch(reason="activity"):
    with _LOCK:
        _STATE["last_activity"] = time.time()
        save_state()
    return reason


# --------------------------------------------------------------------------
# policy wiring
# --------------------------------------------------------------------------


def policy_config():
    merged = dict(_CONFIG.get("policy") or {})
    merged.update(_STATE.get("policy_overrides") or {})
    return PolicyConfig.from_dict(merged)


def snapshot(now=None):
    now = now if now is not None else time.time()
    with _LOCK:
        workloads = [
            Workload(
                id=key,
                started_at=float(value.get("started_at") or now),
                last_heartbeat=float(value.get("last_heartbeat") or now),
                label=value.get("label", ""),
                state=value.get("state", "running"),
            )
            for key, value in (_STATE.get("workloads") or {}).items()
        ]
        starting = bool(_STATE.get("starting"))
        if starting and runtime_alive() is False:
            # A dead model server must not hold the startup grace open.
            starting = False
            _STATE["starting"] = False
            log_event("runtime_exited", pid_path=_CONFIG.get("runtime_pid_path"))
            save_state()
        if starting:
            grace = float(_CONFIG.get("startup_grace_s") or 900.0)
            began = _STATE.get("session_started_at") or _STATE.get("created_at") or now
            if now - float(began) > grace:
                # The CLI never confirmed readiness; stop pretending we booted.
                starting = False
                _STATE["starting"] = False
                log_event("runtime_startup_grace_expired", grace_seconds=grace)
                save_state()
        return Snapshot(
            now=now,
            created_at=float(_STATE.get("created_at") or now),
            last_activity=float(_STATE.get("last_activity") or now),
            workloads=workloads,
            starting=starting,
            locked=bool(_STATE.get("locked")),
            locked_until=_STATE.get("locked_until"),
            provider_state="ready",
            session_started_at=_STATE.get("session_started_at"),
            deadline_at=_STATE.get("deadline_at"),
        )


def runtime_alive():
    """True/False for the model server process; None until it wrote a pid file.

    A pid file older than this container start is a previous session's.
    """
    path = _CONFIG.get("runtime_pid_path")
    if not path:
        return None
    try:
        if os.stat(path).st_mtime < float(_STATE.get("session_started_at") or 0) - 5:
            return None
        with open(path) as handle:
            pid = int(handle.read().strip())
    except (OSError, ValueError):
        return None
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


def tail_lines(path, count):
    """The last ``count`` lines of ``path``, reading backwards from its end."""
    try:
        with open(path, "rb") as handle:
            pos = handle.seek(0, 2)
            data = b""
            while pos > 0 and data.count(b"\n") <= count:
                step = min(65536, pos)
                pos -= step
                handle.seek(pos)
                data = handle.read(step) + data
    except OSError:
        return []
    return data.decode("utf-8", "replace").splitlines()[-count:] if count > 0 else []


def build_context():
    ctx = dict(_CONFIG.get("provider_context") or {})
    ctx["instance_id"] = _STATE.get("instance_id") or _CONFIG.get("instance_id")
    return ctx


def terminate(action, reason, detail=None, force=False):
    """Ask the provider to stop or destroy this machine.

    Returns True on success, False on a provider failure and None when skipped
    (another call in flight, or already terminated and not ``force``). A
    failure never disarms the guard: the loop retries on its next tick.
    """
    detail = detail or {}
    if not _ACTION_LOCK.acquire(blocking=False):
        return None
    try:
        if _STATE.get("terminated") and not force:
            return None
        log_event(
            "lifecycle_guard_triggered",
            action=action,
            reason=reason,
            **detail,
        )
        ctx = build_context()
        try:
            if action == ACTION_DESTROY:
                provider_destroy(ctx)  # noqa: F821 - injected by the builder
                log_event("instance_destroyed", reason=reason, **detail)
            else:
                provider_stop(ctx)  # noqa: F821 - injected by the builder
                log_event("instance_stopped", reason=reason, **detail)
        except Exception as exc:  # noqa: BLE001 - last line of defence
            with _LOCK:
                _STATE["action_failures"] = int(_STATE.get("action_failures") or 0) + 1
                if action == ACTION_STOP:
                    _STATE["stop_failures"] = int(_STATE.get("stop_failures") or 0) + 1
                    _STATE["stop_failing_since"] = _STATE.get("stop_failing_since") or time.time()
                save_state()
            log_event(
                "provider_error",
                action=action,
                reason=reason,
                error=str(exc),
                traceback=traceback.format_exc(limit=3),
            )
            return False
        with _LOCK:
            _STATE["terminated"] = True
            _STATE["terminated_at"] = time.time()
            _STATE["terminated_action"] = action
            _STATE["terminated_reason"] = reason
            _reset_failures()
            # Done: a pending request must not be replayed after a restart.
            _STATE["pending_shutdown"] = None
            save_state()
        return True
    finally:
        _ACTION_LOCK.release()


def reissue_after_s(interval):
    configured = _CONFIG.get("reissue_after_s")
    if configured is not None:
        return float(configured)
    return max(300.0, 5.0 * interval)


def _reset_failures():
    _STATE["stop_failures"] = 0
    _STATE["stop_failing_since"] = None
    _STATE["action_failures"] = 0


def _choose(decision, cfg, now=None):
    """(action, reason, pre-escalation reason) to attempt this tick, or None."""
    action, reason = None, None
    pending = _STATE.get("pending_shutdown")
    if pending:
        # An explicit request outranks whatever the policy would pick.
        action = pending.get("action") or ACTION_STOP
        reason = pending.get("reason") or "manual"
    elif decision.terminates:
        action, reason = decision.action, decision.reason
    if action is None:
        # Back in use: failures from an earlier idle spell must not count
        # towards destroying the disk later.
        if _STATE.get("stop_failures") or _STATE.get("action_failures"):
            _reset_failures()
            save_state()
        return None
    since = _STATE.get("stop_failing_since")
    limit = float(_CONFIG.get("escalate_after_s") or ESCALATE_AFTER_S)
    if (
        action == ACTION_STOP
        and since
        and (now or time.time()) - float(since) >= limit
        and cfg.destroy_supported
    ):
        # Billing forever is worse than losing the disk.
        return ACTION_DESTROY, "stop_failed_escalation", reason
    return action, reason, None


def guard_tick(interval, idle_announced):
    cfg = policy_config()

    with _LOCK:
        if _STATE.get("terminated"):
            since = float(_STATE.get("terminated_at") or 0)
            if time.time() - since <= reissue_after_s(interval):
                return idle_announced
            # The provider accepted the call, yet this machine is still running.
            _STATE["terminated"] = False
            save_state()
            log_event(
                "guard_reissue",
                previous_action=_STATE.get("terminated_action"),
                seconds_since_terminate=round(time.time() - since, 1),
            )

    snap = snapshot()
    with _LOCK:
        if _STATE.get("locked") and snap.lock_expired(cfg):
            _STATE["locked"] = False
            _STATE["locked_until"] = None
            save_state()
            log_event("instance_unlocked", reason="lock_expired")

    decision = evaluate(snap, cfg)
    state = decision.detail.get("state")
    if state == "idle" and not idle_announced:
        idle_announced = True
        log_event("instance_idle", **decision.detail)
    elif state != "idle":
        idle_announced = False

    with _LOCK:
        choice = _choose(decision, cfg)
    for hook in list(TERMINATE_HOOKS.values()):
        if choice is not None:
            try:
                choice = hook(choice, cfg)
            except Exception as exc:  # noqa: BLE001 - a hook never blocks a shutdown
                log_event("watchdog_error", where="terminate_hook", error=str(exc))
    if choice is None:
        return idle_announced
    action, reason, original = choice
    detail = dict(decision.detail)
    if original:
        detail["original_reason"] = original
    if terminate(action, reason, detail) is False:
        log_event(
            "guard_retry",
            action=action,
            reason=reason,
            attempt=int(_STATE.get("action_failures") or 0),
            stop_failures=int(_STATE.get("stop_failures") or 0),
        )
    return idle_announced


def guard_loop():
    interval = float(_CONFIG.get("interval_s") or 60.0)
    cfg = policy_config()
    log_event(
        "watchdog_started",
        interval_seconds=interval,
        idle_timeout_seconds=cfg.idle_timeout_s,
        max_lifetime_seconds=cfg.max_lifetime_s,
        auto_stop=cfg.auto_stop,
        auto_destroy=cfg.auto_destroy,
        session_started_at=_STATE.get("session_started_at"),
    )
    idle_announced = False
    # Only process exit ends this loop: no decision or provider failure may.
    while not _STOPPED.wait(interval):
        try:
            idle_announced = guard_tick(interval, idle_announced)
        except Exception as exc:  # noqa: BLE001 - never let the guard die
            log_event("watchdog_error", error=str(exc), traceback=traceback.format_exc(limit=3))


# --------------------------------------------------------------------------
# inference traffic probe
# --------------------------------------------------------------------------


def _split_sample(line):
    """``name{labels} value [timestamp]`` -> (name, value_text) or None."""
    brace = line.find("{")
    space = line.find(" ")
    if brace != -1 and (space == -1 or brace < space):
        name = line[:brace]
        i, quoted = brace + 1, False
        while i < len(line):
            ch = line[i]
            if quoted and ch == "\\":
                i += 2
                continue
            if ch == '"':
                quoted = not quoted
            elif ch == "}" and not quoted:
                break
            i += 1
        rest = line[i + 1 :].split()
    else:
        parts = line.split()
        name, rest = (parts[0], parts[1:]) if parts else ("", [])
    if not name or not rest:
        return None
    return name, rest[0]


def parse_prometheus(text):
    """Prometheus text format -> {metric name: sum of samples across label sets}."""
    totals = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sample = _split_sample(line)
        if sample is None:
            continue
        name, value_text = sample
        try:
            value = float(value_text)
        except ValueError:
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def metric_value(totals, name):
    """Look a metric up, tolerating exporters that add or drop ``_total``."""
    if name in totals:
        return totals[name]
    alt = name[: -len("_total")] if name.endswith("_total") else name + "_total"
    return totals.get(alt)


def traffic_seen(totals, previous, gauges, counters):
    """True when a gauge is non-zero or a counter grew since ``previous``."""
    for name in gauges:
        value = metric_value(totals, name)
        if value is not None and value > 0:
            return True
    if previous is None:
        return False  # the first read only establishes the baseline
    for name in counters:
        value = metric_value(totals, name)
        before = metric_value(previous, name)
        # A drop means the runtime restarted its counters: rebaseline, not traffic.
        if value is not None and before is not None and value > before:
            return True
    return False


def probe_once(spec, previous):
    """One scrape. Returns the new baseline (the old one on error)."""
    now = time.time()
    try:
        with urllib.request.urlopen(spec["url"], timeout=5) as response:
            text = response.read().decode("utf-8", "replace")
        totals = parse_prometheus(text)
    except Exception as exc:  # noqa: BLE001 - a probe error is never activity
        with _LOCK:
            _PROBE.update({"ok": False, "at": now, "error": str(exc)})
            last_logged = _PROBE.get("error_logged_at") or 0
            if now - last_logged >= PROBE_ERROR_LOG_EVERY_S:
                _PROBE["error_logged_at"] = now
                log_event("activity_probe_error", url=spec.get("url"), error=str(exc))
        return previous

    gauges = list(spec.get("gauges") or [])
    counters = list(spec.get("counters") or [])
    traffic = traffic_seen(totals, previous, gauges, counters)
    summary = {}
    for name in gauges + counters:
        value = metric_value(totals, name)
        if value is not None:
            summary[name] = value
    with _LOCK:
        _PROBE.update({"ok": True, "at": now, "error": None, "values": summary})
        if traffic:
            _PROBE["last_traffic_at"] = now
        # The server answers only once the model is loaded. Without this the
        # startup grace (most of an hour) held the idle clock whenever the
        # CLI's /ready never arrived, e.g. after a restart it did not follow.
        ready = bool(_STATE.get("starting"))
        if ready:
            _STATE["starting"] = False
            _STATE["last_activity"] = now
            save_state()
    if ready:
        log_event("runtime_ready", source="activity_probe")
    if traffic:
        touch("inference_traffic")
    return totals


def probe_loop(spec):
    interval = float(_CONFIG.get("interval_s") or 60.0)
    period = min(interval, 15.0)
    previous = None
    while not _STOPPED.is_set():
        try:
            previous = probe_once(spec, previous)
        except Exception as exc:  # noqa: BLE001
            log_event("watchdog_error", where="activity_probe", error=str(exc))
        _STOPPED.wait(period)


def probe_spec():
    spec = _CONFIG.get("activity_probe")
    if not isinstance(spec, dict) or not spec.get("url"):
        return None
    if (spec.get("kind") or "prometheus") != "prometheus":
        return None
    return spec


def probe_payload():
    spec = probe_spec()
    if spec is None:
        return None
    with _LOCK:
        return {
            "url": spec.get("url"),
            "ok": _PROBE.get("ok"),
            "at": _PROBE.get("at"),
            "error": _PROBE.get("error"),
            "values": dict(_PROBE.get("values") or {}),
            "last_traffic_at": _PROBE.get("last_traffic_at"),
        }


# --------------------------------------------------------------------------
# extensions
# --------------------------------------------------------------------------


def load_extension(name, source):
    exec(compile(source, "<extension %s>" % name, "exec"), globals())
    _EXT[name] = hashlib.sha256(source.encode()).hexdigest()[:16]
    log_event("extension_loaded", name=name, digest=_EXT[name])


def load_saved_extensions():
    try:
        names = sorted(os.listdir(EXT_DIR))
    except OSError:
        return
    for filename in names:
        if not filename.endswith(".py"):
            continue
        try:
            with open(os.path.join(EXT_DIR, filename)) as handle:
                load_extension(filename[:-3], handle.read())
        except Exception as exc:  # noqa: BLE001 - a bad extension must not kill the guard
            log_event("extension_error", name=filename, error=str(exc))


def extend(body):
    name = str(body.get("name") or "")
    if not name.isidentifier():
        return 400, {"error": "bad extension name"}
    source = zlib.decompress(base64.b64decode(body.get("source") or "")).decode()
    os.makedirs(EXT_DIR, exist_ok=True)
    _atomic_write(os.path.join(EXT_DIR, name + ".py"), source)
    load_extension(name, source)
    return 200, {"ok": True, "extensions": dict(_EXT)}


def _ext_route(request, method, path, body=None):
    for handler in list(EXT_ROUTES.values()):
        if handler(request, method, path, body):
            return True
    return False


# --------------------------------------------------------------------------
# HTTP control surface
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "jambu-watchdog/1.0"

    def log_message(self, fmt, *args):  # noqa: A003 - silence access logs
        return

    # -- plumbing -------------------------------------------------------

    def _authorized(self):
        expected = _CONFIG.get("token")
        if not expected:
            return True
        supplied = self.headers.get("X-Jambu-Token") or ""
        if not hmac.compare_digest(supplied.encode(), str(expected).encode()):
            self._reply(401, {"error": "unauthorized"})
            return False
        return True

    def _reply(self, code, payload):
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def _payload(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    # -- routes ---------------------------------------------------------

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/health":
            self._reply(200, {"ok": True, "service": "jambu-watchdog"})
            return
        if not self._authorized():
            return
        if path == "/state":
            self._reply(200, state_payload())
            return
        if path == "/events":
            try:
                limit = int(parse_qs(self.path.partition("?")[2])["limit"][0])
            except (KeyError, ValueError):
                limit = 100
            self._reply(200, {"events": read_events(limit)})
            return
        if path == "/logs":
            query = parse_qs(self.path.partition("?")[2])
            name = (query.get("name") or ["runtime"])[0]
            target = (_CONFIG.get("logs") or {}).get(name)
            if not target:
                self._reply(404, {"error": "unknown log %r" % name})
                return
            try:
                count = max(0, min(int(query["lines"][0]), LOG_LINES_MAX))
            except (KeyError, ValueError):
                count = 200
            self._reply(200, {"name": name, "lines": tail_lines(target, count)})
            return
        if self._extended("GET", path):
            return
        self._reply(404, {"error": "not found"})

    def do_PUT(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if self._authorized() and not self._extended("PUT", path):
            self._reply(404, {"error": "not found"})

    def _extended(self, method, path, body=None):
        try:
            return _ext_route(self, method, path, body)
        except Exception as exc:  # noqa: BLE001
            log_event("watchdog_error", route=path, error=str(exc))
            self._reply(500, {"error": str(exc)})
            return True

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if not self._authorized():
            return
        body = self._payload()
        if self._extended("POST", path, body):
            return
        try:
            self._reply(*dispatch(path, body))
        except Exception as exc:  # noqa: BLE001
            log_event("watchdog_error", route=path, error=str(exc))
            self._reply(500, {"error": str(exc)})


def state_payload():
    cfg = policy_config()
    snap = snapshot()
    decision = evaluate(snap, cfg)
    with _LOCK:
        workloads = json.loads(json.dumps(_STATE.get("workloads") or {}, default=str))
        return {
            "instance_id": _STATE.get("instance_id") or _CONFIG.get("instance_id"),
            "provider": _CONFIG.get("provider"),
            "created_at": _STATE.get("created_at"),
            "last_activity": _STATE.get("last_activity"),
            "starting": snap.starting,
            "runtime_alive": runtime_alive(),
            "locked": snap.lock_active(cfg),
            "locked_until": _STATE.get("locked_until"),
            "deadline_at": _STATE.get("deadline_at"),
            "terminated": bool(_STATE.get("terminated")),
            "session_started_at": _STATE.get("session_started_at"),
            "session_age_seconds": round(snap.session_age_seconds(), 1),
            "stop_failures": int(_STATE.get("stop_failures") or 0),
            "pending_shutdown": _STATE.get("pending_shutdown"),
            "last_probe": probe_payload(),
            "extensions": dict(_EXT),
            "state": derive_state(snap, cfg),
            "idle_seconds": round(snap.idle_seconds(cfg), 1),
            "age_seconds": round(snap.age_seconds(), 1),
            "workloads": workloads,
            "decision": decision.to_dict(),
            "policy": cfg.to_dict(),
            "now": time.time(),
        }


def read_events(limit=100):
    try:
        with open(EVENTS_PATH) as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def dispatch(path, body):
    now = time.time()

    if path == "/identify":
        # The instance id only exists after creation, and the onstart script is
        # baked before that; the CLI tells the watchdog who it is once it can.
        with _LOCK:
            if body.get("instance_id"):
                _STATE["instance_id"] = str(body["instance_id"])
            overrides = _STATE.setdefault("policy_overrides", {})
            if body.get("hourly_cost_usd") is not None:
                overrides["hourly_cost_usd"] = float(body["hourly_cost_usd"])
            if body.get("policy"):
                overrides.update(dict(body["policy"]))
            _STATE["last_activity"] = now
            save_state()
        log_event("watchdog_identified", instance_id=_STATE.get("instance_id"))
        return 200, {"ok": True, "instance_id": _STATE.get("instance_id")}

    if path == "/activity":
        touch()
        return 200, {"ok": True, "last_activity": _STATE["last_activity"]}

    if path == "/ready":
        with _LOCK:
            _STATE["starting"] = False
            _STATE["last_activity"] = now
            save_state()
        log_event("runtime_ready")
        return 200, {"ok": True}

    if path == "/starting":
        with _LOCK:
            _STATE["starting"] = True
            _STATE["created_at"] = _STATE.get("created_at") or now
            _STATE["last_activity"] = now
            save_state()
        log_event("runtime_starting")
        return 200, {"ok": True}

    if path == "/workload/start":
        workload_id = str(body.get("id") or "w-%d" % int(now * 1000))
        with _LOCK:
            _STATE.setdefault("workloads", {})[workload_id] = {
                "started_at": now,
                "last_heartbeat": now,
                "label": body.get("label", ""),
                "state": body.get("state", "running"),
            }
            _STATE["last_activity"] = now
            _STATE["starting"] = False
            save_state()
        log_event("workload_started", workload_id=workload_id, label=body.get("label", ""))
        return 200, {"ok": True, "id": workload_id}

    if path == "/workload/heartbeat":
        workload_id = body.get("id")
        with _LOCK:
            record = (_STATE.get("workloads") or {}).get(workload_id)
            if record is None and workload_id:
                record = {
                    "started_at": now,
                    "label": body.get("label", ""),
                    "state": "running",
                }
                _STATE.setdefault("workloads", {})[workload_id] = record
            if record is not None:
                record["last_heartbeat"] = now
            _STATE["last_activity"] = now
            save_state()
        return 200, {"ok": True, "id": workload_id}

    if path == "/workload/end":
        workload_id = body.get("id")
        with _LOCK:
            record = (_STATE.get("workloads") or {}).get(workload_id)
            if record is not None:
                record["state"] = body.get("state") or (
                    "finished" if not body.get("returncode") else "failed"
                )
                record["finished_at"] = now
                record["returncode"] = body.get("returncode")
            _STATE["last_activity"] = now
            save_state()
        log_event(
            "workload_finished",
            workload_id=workload_id,
            returncode=body.get("returncode"),
        )
        return 200, {"ok": True}

    if path == "/lock":
        ttl = body.get("ttl_seconds")
        until = body.get("until")
        cfg = policy_config()
        if until is None and ttl:
            until = now + float(ttl)
        if until is None and not cfg.allow_indefinite_lock:
            return 400, {"error": "indefinite locks are disabled"}
        with _LOCK:
            _STATE["locked"] = True
            _STATE["locked_until"] = float(until) if until is not None else None
            _STATE["last_activity"] = now
            save_state()
        log_event("instance_locked", locked_until=until, reason=body.get("reason", ""))
        return 200, {"ok": True, "locked_until": until}

    if path == "/deadline":
        try:
            until = float(body.get("until"))
        except (TypeError, ValueError):
            return 400, {"error": "bad until"}
        with _LOCK:
            old = _STATE.get("deadline_at") or 0
            keep = _STATE["deadline_at"] = max(old, until)
            save_state()
        if keep != old:
            log_event("pool_deadline_extended", deadline_at=keep)
        return 200, {"ok": True, "deadline_at": keep}

    if path == "/unlock":
        with _LOCK:
            _STATE["locked"] = False
            _STATE["locked_until"] = None
            _STATE["last_activity"] = now
            save_state()
        log_event("instance_unlocked", reason=body.get("reason", "manual"))
        return 200, {"ok": True}

    if path == "/extend":
        return extend(body)

    if path == "/shutdown":
        action = body.get("action") or ACTION_STOP
        reason = body.get("reason") or "manual"
        if action not in (ACTION_STOP, ACTION_DESTROY):
            return 400, {"error": "action must be stop or destroy"}
        with _LOCK:
            # Remembered so the guard loop retries it if the provider call fails.
            _STATE["pending_shutdown"] = {"action": action, "reason": reason}
            save_state()
        threading.Thread(
            target=terminate,
            args=(action, reason, {"requested_by": "cli"}),
            kwargs={"force": True},
            daemon=True,
        ).start()
        return 202, {"ok": True, "action": action}

    return 404, {"error": "not found"}


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def main():
    global _CONFIG, _STATE

    try:
        with open(CONFIG_PATH) as handle:
            _CONFIG = json.load(handle)
    except (OSError, ValueError) as exc:
        print("[jambu-watchdog] cannot read %s: %s" % (CONFIG_PATH, exc), file=sys.stderr)
        return 1

    for key, value in (_CONFIG.get("env") or {}).items():
        os.environ.setdefault(str(key), str(value))

    _STATE = load_state()
    save_state()
    load_saved_extensions()

    port = int(_CONFIG.get("port") or 8777)
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True

    guard = threading.Thread(target=guard_loop, daemon=True)
    guard.start()

    spec = probe_spec()
    if spec is not None:
        threading.Thread(target=probe_loop, args=(spec,), daemon=True).start()

    log_event("watchdog_listening", port=port)
    try:
        server.serve_forever(poll_interval=1.0)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        _STOPPED.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
