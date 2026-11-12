"""Redis queue backend: shared by application replicas on different hosts.

Every state change is one Lua script, so the token check, the lease check and the write are
atomic, and every time comparison uses the server's ``TIME`` (never a client clock). Layout under
``<namespace>:``:

    job:<id>      hash of scalar fields (JSON for manifest, limits, outputs_ref, result)
    att:<id>      list of attempts (JSON, one per item)
    keys          hash key -> id (submit idempotency)
    st:<state>    sorted set per state, score = submitted_at (QUEUED is the claim order)
    leased        sorted set of leased ids, score = lease_until
    tokens        hash id -> fencing token of the live lease
    coord         hash owner / lease_until;  paused  string;  drain  string;  audit  stream of control actions

Needs Redis >= 5. Scripts address keys by prefix, so this targets a single Redis (or a primary
with replicas), not a Redis Cluster.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import fields
from typing import Any, Optional

import redis as _redis

# A hung Redis must fail a call, not freeze the coordinator (and its heartbeat) forever.
SOCKET_TIMEOUT_S = 10.0
SOCKET_CONNECT_TIMEOUT_S = 10.0

from ...core.credentials import CredentialResolver
from ...core.errors import ConfigError
from ..backend import QueueBackend
from ..models import FINAL_STATES, Attempt, Claim, JobState, LeaseLost, QueueJob, QueueStats

_FIELDS = {f.name for f in fields(QueueJob)} - {"id", "key", "submitted_at"}
_LEASED_STATES = frozenset({JobState.CLAIMED, JobState.RUNNING, JobState.DELIVERING})
_JSON_FIELDS = ("manifest", "limits", "outputs_ref", "result")
_ALL_STATES = [s.value for s in JobState]

_PRELUDE = """
if redis.replicate_commands then redis.replicate_commands() end
local P = ARGV[1]
local OFF = tonumber(ARGV[2])
local tm = redis.call('TIME')
local NOW = tonumber(tm[1]) + tonumber(tm[2]) / 1000000 + OFF
local function f(x) return string.format('%.6f', x) end
local function jk(id) return P .. ':job:' .. id end
local function stk(s) return P .. ':st:' .. s end
local function setstate(id, new)
  local old = redis.call('HGET', jk(id), 'state')
  if old then redis.call('ZREM', stk(old), id) end
  redis.call('HSET', jk(id), 'state', new)
  redis.call('ZADD', stk(new), redis.call('HGET', jk(id), 'submitted_at'), id)
end
local function ret(status, id)
  return {status, redis.call('HGETALL', jk(id)), redis.call('LRANGE', P .. ':att:' .. id, 0, -1)}
end
local function release(id)
  redis.call('HDEL', P .. ':tokens', id)
  redis.call('ZREM', P .. ':leased', id)
end
local function audit(actor, action, job_id, detail)
  redis.call('XADD', P .. ':audit', 'MAXLEN', '~', '10000', '*', 'ts', f(NOW), 'actor', actor, 'action', action,
             'job_id', job_id, 'detail', detail)
end
local function reap()
  local ids = redis.call('ZRANGEBYSCORE', P .. ':leased', '-inf', f(NOW))
  for _, id in ipairs(ids) do
    release(id)
    if redis.call('EXISTS', jk(id)) == 0 then
      for _, s in ipairs({'queued','claimed','running','delivering','succeeded','failed','cancelled'}) do
        redis.call('ZREM', stk(s), id)
      end
    else
      local nw = 'queued'
      if redis.call('HGET', jk(id), 'cancel_requested') == '1' then nw = 'cancelled' end
      redis.call('HDEL', jk(id), 'machine')
      setstate(id, nw)
    end
  end
end
local function owned(id, token)
  local tk = redis.call('HGET', P .. ':tokens', id)
  return tk ~= false and tk == token
end
reap()
"""

_SCRIPTS = {
    "reap": "return 1",
    # ARGV: key, id, then field/value pairs for the new job's hash.
    "submit": """
local key, id = ARGV[3], ARGV[4]
local existing = redis.call('HGET', P .. ':keys', key)
if existing then return ret('EXISTS', existing) end
if redis.call('EXISTS', jk(id)) == 1 then return {'IDCONFLICT'} end
redis.call('HSET', P .. ':keys', key, id)
local natt = tonumber(ARGV[5])
for i = 6, 5 + natt do redis.call('RPUSH', P .. ':att:' .. id, ARGV[i]) end
for i = 6 + natt, #ARGV, 2 do redis.call('HSET', jk(id), ARGV[i], ARGV[i + 1]) end
redis.call('HSET', jk(id), 'submitted_at', f(NOW))
local st = redis.call('HGET', jk(id), 'state')
redis.call('HDEL', jk(id), 'state')
setstate(id, st)
return ret('OK', id)
""",
    # ARGV: lease_s, token, job_id ('' = oldest)
    "claim": """
local lease_s, token, want = tonumber(ARGV[3]), ARGV[4], ARGV[5]
-- a paused queue still lets a coordinator re-adopt one named job
if want == '' and redis.call('GET', P .. ':paused') == '1' then return {'NONE'} end
local pick = nil
if want ~= '' then
  if redis.call('ZSCORE', stk('queued'), want) and redis.call('HGET', jk(want), 'cancel_requested') ~= '1' then
    pick = want
  end
else
  local ids = redis.call('ZRANGE', stk('queued'), 0, 99)
  for _, id in ipairs(ids) do
    if redis.call('HGET', jk(id), 'cancel_requested') ~= '1' then pick = id; break end
  end
end
if not pick then return {'NONE'} end
setstate(pick, 'claimed')
redis.call('HSET', P .. ':tokens', pick, token)
redis.call('ZADD', P .. ':leased', f(NOW + lease_s), pick)
local r = ret('OK', pick)
r[4] = f(NOW + lease_s)
return r
""",
    # ARGV: id, token, lease_s
    "renew": """
local id, token, lease_s = ARGV[3], ARGV[4], tonumber(ARGV[5])
if not owned(id, token) then return {'LEASE_LOST'} end
redis.call('ZADD', P .. ':leased', f(NOW + lease_s), id)
local r = ret('OK', id)
r[4] = f(NOW + lease_s)
return r
""",
    # ARGV: id, token, attempts_mode (-1 keep | n replace with n items), items..., nfields, (name, kind, value)*
    "update": """
local id, token = ARGV[3], ARGV[4]
if not owned(id, token) then return {'LEASE_LOST'} end
local i = 5
local n = tonumber(ARGV[i]); i = i + 1
if n >= 0 then
  redis.call('DEL', P .. ':att:' .. id)
  for _ = 1, n do redis.call('RPUSH', P .. ':att:' .. id, ARGV[i]); i = i + 1 end
end
local nf = tonumber(ARGV[i]); i = i + 1
for _ = 1, nf do
  local name, kind, value = ARGV[i], ARGV[i + 1], ARGV[i + 2]
  i = i + 3
  if kind == 'n' then redis.call('HDEL', jk(id), name)
  elseif name == 'state' then setstate(id, value)
  else redis.call('HSET', jk(id), name, value) end
end
return ret('OK', id)
""",
    # ARGV: id, token, attempt_json
    "requeue": """
local id, token = ARGV[3], ARGV[4]
if not owned(id, token) then return {'LEASE_LOST'} end
redis.call('RPUSH', P .. ':att:' .. id, ARGV[5])
release(id)
local nw = 'queued'
if redis.call('HGET', jk(id), 'cancel_requested') == '1' then nw = 'cancelled' end
redis.call('HDEL', jk(id), 'machine')
setstate(id, nw)
return ret('OK', id)
""",
    # ARGV: id, token, state, result_json
    "settle": """
local id, token = ARGV[3], ARGV[4]
if not owned(id, token) then return {'LEASE_LOST'} end
release(id)
redis.call('HSET', jk(id), 'result', ARGV[6])
setstate(id, ARGV[5])
return ret('OK', id)
""",
    # ARGV: id, actor
    "cancel": """
local id, actor = ARGV[3], ARGV[4]
if redis.call('EXISTS', jk(id)) == 0 then return {'MISSING'} end
audit(actor, 'request_cancel', id, '')
local st = redis.call('HGET', jk(id), 'state')
if st == 'succeeded' or st == 'failed' or st == 'cancelled' then return ret('OK', id) end
if st == 'queued' then setstate(id, 'cancelled') end
redis.call('HSET', jk(id), 'cancel_requested', '1')
return ret('OK', id)
""",
    # ARGV: id, actor
    "retry": """
local id, actor = ARGV[3], ARGV[4]
if redis.call('EXISTS', jk(id)) == 0 then return {'MISSING'} end
local st = redis.call('HGET', jk(id), 'state')
if st ~= 'failed' then return {'BADSTATE', st} end
redis.call('HSET', jk(id), 'cancel_requested', '0')
redis.call('HDEL', jk(id), 'machine')
setstate(id, 'queued')
audit(actor, 'retry', id, '')
return ret('OK', id)
""",
    # ARGV: paused ('1'/'0'), actor
    "pause": """
redis.call('SET', P .. ':paused', ARGV[3])
audit(ARGV[4], 'set_paused', '', ARGV[3] == '1' and 'true' or 'false')
return 1
""",
    # ARGV: '1' (request) / '0' (clear), actor
    "drain": """
if ARGV[3] == '1' then
  redis.call('SET', P .. ':drain', '1')
  audit(ARGV[4], 'request_drain', '', '')
else
  redis.call('DEL', P .. ':drain')
end
return 1
""",
    # returns counts per state (in JobState order), then oldest queued score and now
    "stats": """
local r = {}
for _, s in ipairs({'queued','claimed','running','delivering','succeeded','failed','cancelled'}) do
  r[#r + 1] = redis.call('ZCARD', stk(s))
end
local o = redis.call('ZRANGE', stk('queued'), 0, 0, 'WITHSCORES')
r[#r + 1] = o[2] or ''
r[#r + 1] = f(NOW)
return r
""",
    # ARGV: owner, lease_s, mode ('acquire' | 'renew')
    "coord_take": """
local owner, lease_s, mode = ARGV[3], tonumber(ARGV[4]), ARGV[5]
local cur = redis.call('HGET', P .. ':coord', 'owner')
local lease = tonumber(redis.call('HGET', P .. ':coord', 'lease') or '0')
local live = cur ~= false and lease > NOW
if mode == 'acquire' then
  if live and cur ~= owner then return 0 end
else
  if not live or cur ~= owner then return 0 end
end
redis.call('HSET', P .. ':coord', 'owner', owner, 'lease', f(NOW + lease_s))
return 1
""",
    "coord_release": """
local cur = redis.call('HGET', P .. ':coord', 'owner')
local lease = tonumber(redis.call('HGET', P .. ':coord', 'lease') or '0')
if cur == ARGV[3] and lease > NOW then redis.call('DEL', P .. ':coord') end
return 1
""",
    # ARGV: n, then state names (n = 0 means all), limit. A consistent snapshot of the matching jobs.
    "list": """
local n = tonumber(ARGV[3])
local limit = tonumber(ARGV[4 + n])
local names = {}
if n == 0 then names = {'queued','claimed','running','delivering','succeeded','failed','cancelled'}
else for i = 1, n do names[i] = ARGV[3 + i] end end
local rows = {}
for _, s in ipairs(names) do
  local z = redis.call('ZRANGE', stk(s), 0, limit - 1, 'WITHSCORES')
  for i = 1, #z, 2 do rows[#rows + 1] = {tonumber(z[i + 1]), z[i]} end
end
table.sort(rows, function(a, b) if a[1] ~= b[1] then return a[1] < b[1] end return a[2] < b[2] end)
local out = {}
for i = 1, math.min(limit, #rows) do out[#out + 1] = ret('OK', rows[i][2]) end
return out
""",
    "get": """
if redis.call('EXISTS', jk(ARGV[3])) == 0 then return {'NONE'} end
return ret('OK', ARGV[3])
""",
    "coord_get": """
local cur = redis.call('HGET', P .. ':coord', 'owner')
local lease = tonumber(redis.call('HGET', P .. ':coord', 'lease') or '0')
if cur ~= false and lease > NOW then return {cur, f(lease)} end
return {}
""",
}


def _encode(name: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    if name in _JSON_FIELDS:
        return json.dumps(value)
    if name == "cancel_requested":
        return "1" if value else "0"
    if name == "state":
        return JobState(value).value
    return str(value)


def _attempt_json(a: Any) -> str:
    return json.dumps(a.to_dict() if isinstance(a, Attempt) else a)


def _job(raw: list) -> QueueJob:
    flat, att = raw
    h = dict(zip(flat[0::2], flat[1::2]))
    d: dict[str, Any] = {
        "id": h["id"], "key": h["key"], "bundle_ref": h.get("bundle_ref", ""),
        "submitted_at": float(h["submitted_at"]), "submitted_by": h.get("submitted_by", ""),
        "state": h["state"], "cancel_requested": h.get("cancel_requested") == "1",
        "machine": h.get("machine"),
        "attempts": [json.loads(a) for a in att],
    }
    d["manifest"] = json.loads(h["manifest"]) if "manifest" in h else {}
    for name in ("limits", "outputs_ref", "result"):
        d[name] = json.loads(h[name]) if name in h else None
    return QueueJob.from_dict(d)


class RedisBackend(QueueBackend):
    name = "redis"

    def __init__(self, url: str, namespace: str, client: Any = None) -> None:
        self.namespace = namespace
        self._test_clock_offset = 0.0  # test-only: shifts the server clock; production leaves it 0
        self._r = client or _redis.Redis.from_url(
            url, decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT_S, socket_connect_timeout=SOCKET_CONNECT_TIMEOUT_S,
        )
        self._scripts = {n: self._r.register_script(_PRELUDE + body) for n, body in _SCRIPTS.items()}

    def __repr__(self) -> str:
        return f"RedisBackend(namespace={self.namespace!r})"

    def _run(self, name: str, *args: Any) -> Any:
        return self._scripts[name](keys=[], args=[self.namespace, repr(self._test_clock_offset), *args])

    def now(self) -> float:
        sec, usec = self._r.time()
        return float(sec) + float(usec) / 1_000_000 + self._test_clock_offset

    # -- contract ----------------------------------------------------------------
    def submit(self, job: QueueJob) -> QueueJob:
        pairs: list[str] = []
        for name, value in job.to_dict().items():
            if name in ("key", "submitted_at", "attempts"):
                continue
            enc = _encode(name, value)
            if enc is not None:
                pairs += [name, enc]
        atts = [_attempt_json(a) for a in job.attempts]
        res = self._run("submit", job.key, job.id, str(len(atts)), *atts, *pairs, "key", job.key)
        if res[0] == "IDCONFLICT":
            raise ValueError(f"job id {job.id} already exists under a different key")
        return _job(res[1:3])

    def claim(self, worker: str, lease_s: float, job_id: Optional[str] = None) -> Optional[Claim]:
        token = secrets.token_hex(16)
        res = self._run("claim", repr(float(lease_s)), token, job_id or "")
        if res[0] != "OK":
            return None
        return Claim(job=_job(res[1:3]), token=token, lease_until=float(res[3]))

    def renew(self, claim: Claim, lease_s: float) -> Claim:
        res = self._run("renew", claim.job.id, claim.token, repr(float(lease_s)))
        self._check(res, claim)
        return Claim(job=_job(res[1:3]), token=claim.token, lease_until=float(res[3]))

    def update(self, claim: Claim, **updates: Any) -> QueueJob:
        """Updatable: every QueueJob field except id, key and submitted_at. ``state`` may only be
        CLAIMED, RUNNING or DELIVERING (QUEUED goes through requeue, final states through settle)."""
        bad = set(updates) - _FIELDS
        if bad:
            raise ValueError(f"cannot update fields: {sorted(bad)}")
        if "state" in updates:
            updates["state"] = JobState(updates["state"])
            if updates["state"] not in _LEASED_STATES:
                raise ValueError(f"update cannot set state {updates['state'].value}; use requeue or settle")
        att = updates.pop("attempts", None)
        att_args = ["-1"] if att is None else [str(len(att)), *[_attempt_json(a) for a in att]]
        triples: list[str] = []
        for name, value in updates.items():
            enc = _encode(name, value)
            triples += [name, "n", ""] if enc is None else [name, "v", enc]
        res = self._run("update", claim.job.id, claim.token, *att_args, str(len(triples) // 3), *triples)
        self._check(res, claim)
        return _job(res[1:3])

    def requeue(self, claim: Claim, attempt: Attempt) -> QueueJob:
        res = self._run("requeue", claim.job.id, claim.token, _attempt_json(attempt))
        self._check(res, claim)
        return _job(res[1:3])

    def settle(self, claim: Claim, state: JobState, result: dict) -> QueueJob:
        state = JobState(state)
        if state not in FINAL_STATES:
            raise ValueError(f"settle needs a final state, got {state.value}")
        res = self._run("settle", claim.job.id, claim.token, state.value, json.dumps(result))
        self._check(res, claim)
        return _job(res[1:3])

    @staticmethod
    def _check(res: list, claim: Claim) -> None:
        if res[0] != "OK":
            raise LeaseLost(f"lease lost for job {claim.job.id}")

    def request_cancel(self, job_id: str, actor: str) -> QueueJob:
        res = self._run("cancel", job_id, actor)
        if res[0] != "OK":
            raise KeyError(job_id)
        return _job(res[1:3])

    def retry(self, job_id: str, actor: str) -> QueueJob:
        res = self._run("retry", job_id, actor)
        if res[0] == "MISSING":
            raise KeyError(job_id)
        if res[0] == "BADSTATE":
            raise ValueError(f"job {job_id} is {res[1]}, only failed jobs can be retried")
        return _job(res[1:3])

    def get(self, job_id: str) -> Optional[QueueJob]:
        res = self._run("get", job_id)
        return _job(res[1:3]) if res[0] == "OK" else None

    def list(self, states: Optional[set[JobState]] = None, limit: int = 100) -> list[QueueJob]:
        wanted = [JobState(s).value for s in states] if states else []
        if limit < 1:
            return []
        res = self._run("list", str(len(wanted)), *wanted, str(limit))
        return [_job(r[1:3]) for r in res]

    def stats(self) -> QueueStats:
        res = self._run("stats")
        by_state = {s: int(n) for s, n in zip(_ALL_STATES, res[:7]) if int(n)}
        oldest = res[7]
        return QueueStats(by_state, None if oldest in ("", None) else float(res[8]) - float(oldest))

    def acquire_coordinator(self, owner: str, lease_s: float) -> bool:
        return bool(self._run("coord_take", owner, repr(float(lease_s)), "acquire"))

    def renew_coordinator(self, owner: str, lease_s: float) -> bool:
        return bool(self._run("coord_take", owner, repr(float(lease_s)), "renew"))

    def release_coordinator(self, owner: str) -> None:
        self._run("coord_release", owner)

    def coordinator(self) -> Optional[dict]:
        res = self._run("coord_get")
        return {"owner": res[0], "lease_until": float(res[1])} if res else None

    def set_paused(self, paused: bool, actor: str) -> None:
        self._run("pause", "1" if paused else "0", actor)

    def paused(self) -> bool:
        return self._r.get(f"{self.namespace}:paused") == "1"

    def request_drain(self, actor: str) -> None:
        self._run("drain", "1", actor)

    def drain_requested(self) -> bool:
        return self._r.get(f"{self.namespace}:drain") == "1"

    def clear_drain(self) -> None:
        self._run("drain", "0", "")

    def audit(self) -> list[dict]:
        rows = self._r.xrange(f"{self.namespace}:audit")
        return [
            {"ts": float(f["ts"]), "actor": f["actor"], "action": f["action"],
             "job_id": f["job_id"] or None, "detail": f["detail"]}
            for _, f in rows
        ]


_NAMESPACE_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _namespace(config: Any, resolver: CredentialResolver) -> str:
    queue = getattr(config, "queue", None)
    name = getattr(queue, "namespace", None) or os.environ.get("GPU_QUEUE_NAMESPACE") or ""
    if name and not _NAMESPACE_RE.fullmatch(name):
        raise ConfigError("queue namespace must match [A-Za-z0-9_.-] and be 1-64 characters")
    return "gpu4agents:" + (name or "fp-" + config.fingerprint()[:12])


def create(config: Any) -> RedisBackend:
    provider = getattr(getattr(config, "provider", None), "profile", "default")
    resolver = CredentialResolver(getattr(config, "project_dir", None), profile=provider)
    url = resolver.get("GPU_QUEUE_URL")
    if not url:
        raise ConfigError(
            "queue backend redis needs GPU_QUEUE_URL (a redis:// or rediss:// address) in the "
            "environment or ~/.gpu/credentials; it is never read from gpu.yaml"
        )
    return RedisBackend(url, _namespace(config, resolver))
