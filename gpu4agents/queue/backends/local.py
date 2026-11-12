"""Local queue backend on SQLite (WAL). Shared by processes on one machine."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from ..backend import QueueBackend
from ..models import FINAL_STATES, Attempt, Claim, JobState, LeaseLost, QueueJob, QueueStats

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    payload TEXT NOT NULL,
    token TEXT,
    lease_until REAL,
    submitted_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, submitted_at);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, job_id TEXT, detail TEXT
);
"""

_LEASED = (JobState.CLAIMED.value, JobState.RUNNING.value, JobState.DELIVERING.value)
_FIELDS = {f.name for f in fields(QueueJob)} - {"id", "key"}


class LocalBackend(QueueBackend):
    name = "local"

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(_SCHEMA)

    def now(self) -> float:
        return self._clock()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                self._reap(c)
                yield c
            except BaseException:
                c.execute("ROLLBACK")
                raise
            else:
                c.execute("COMMIT")

    # -- helpers -----------------------------------------------------------------
    def _reap(self, c: sqlite3.Connection) -> None:
        now = self.now()
        marks = ",".join("?" * len(_LEASED))
        rows = c.execute(
            f"SELECT id, payload FROM jobs WHERE state IN ({marks}) AND lease_until <= ?", (*_LEASED, now)
        ).fetchall()
        for r in rows:
            job = QueueJob.from_dict(json.loads(r["payload"]))
            job.state = JobState.CANCELLED if job.cancel_requested else JobState.QUEUED
            job.machine = None
            self._save(c, job, token=None, lease_until=None)

    @staticmethod
    def _save(c: sqlite3.Connection, job: QueueJob, token: Optional[str], lease_until: Optional[float]) -> None:
        c.execute(
            "UPDATE jobs SET state=?, payload=?, token=?, lease_until=? WHERE id=?",
            (job.state.value, json.dumps(job.to_dict()), token, lease_until, job.id),
        )

    @staticmethod
    def _job(row: sqlite3.Row) -> QueueJob:
        return QueueJob.from_dict(json.loads(row["payload"]))

    def _owned(self, c: sqlite3.Connection, claim: Claim) -> tuple[QueueJob, sqlite3.Row]:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (claim.job.id,)).fetchone()
        if row is None or row["token"] is None or row["token"] != claim.token or row["lease_until"] <= self.now():
            raise LeaseLost(f"lease lost for job {claim.job.id}")
        return self._job(row), row

    def _audit(self, c: sqlite3.Connection, actor: str, action: str, job_id: Optional[str], detail: str = "") -> None:
        c.execute(
            "INSERT INTO audit (ts, actor, action, job_id, detail) VALUES (?,?,?,?,?)",
            (self.now(), actor, action, job_id, detail),
        )

    def audit(self) -> list[dict]:
        with self._tx() as c:
            return [dict(r) for r in c.execute("SELECT ts, actor, action, job_id, detail FROM audit ORDER BY id")]

    # -- contract ----------------------------------------------------------------
    def submit(self, job: QueueJob) -> QueueJob:
        with self._tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE key=?", (job.key,)).fetchone()
            if row is not None:
                return self._job(row)
            job.submitted_at = self.now()
            c.execute(
                "INSERT INTO jobs (id, key, state, payload, submitted_at) VALUES (?,?,?,?,?)",
                (job.id, job.key, job.state.value, json.dumps(job.to_dict()), job.submitted_at),
            )
            return job

    def claim(self, worker: str, lease_s: float, job_id: Optional[str] = None) -> Optional[Claim]:
        with self._tx() as c:
            if self._meta(c, "paused") == "1":
                return None
            rows = c.execute(
                "SELECT * FROM jobs WHERE state=? " + ("AND id=? " if job_id else "") + "ORDER BY submitted_at, seq",
                (JobState.QUEUED.value, *([job_id] if job_id else [])),
            ).fetchall()
            for row in rows:
                job = self._job(row)
                if job.cancel_requested:
                    continue
                job.state = JobState.CLAIMED
                token = secrets.token_hex(16)
                until = self.now() + lease_s
                self._save(c, job, token, until)
                return Claim(job=job, token=token, lease_until=until)
            return None

    def renew(self, claim: Claim, lease_s: float) -> Claim:
        with self._tx() as c:
            job, _ = self._owned(c, claim)
            until = self.now() + lease_s
            self._save(c, job, claim.token, until)
            return Claim(job=job, token=claim.token, lease_until=until)

    def update(self, claim: Claim, **updates: Any) -> QueueJob:
        bad = set(updates) - _FIELDS
        if bad:
            raise ValueError(f"cannot update fields: {sorted(bad)}")
        with self._tx() as c:
            job, row = self._owned(c, claim)
            for k, v in updates.items():
                setattr(job, k, JobState(v) if k == "state" else v)
            self._save(c, job, row["token"], row["lease_until"])
            return job

    def requeue(self, claim: Claim, attempt: Attempt) -> QueueJob:
        with self._tx() as c:
            job, _ = self._owned(c, claim)
            job.attempts.append(attempt)
            job.state = JobState.CANCELLED if job.cancel_requested else JobState.QUEUED
            job.machine = None
            self._save(c, job, None, None)
            return job

    def settle(self, claim: Claim, state: JobState, result: dict) -> QueueJob:
        state = JobState(state)
        if state not in FINAL_STATES:
            raise ValueError(f"settle needs a final state, got {state.value}")
        with self._tx() as c:
            job, _ = self._owned(c, claim)
            job.state = state
            job.result = result
            self._save(c, job, None, None)
            return job

    def request_cancel(self, job_id: str, actor: str) -> QueueJob:
        with self._tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            job = self._job(row)
            self._audit(c, actor, "request_cancel", job_id)
            if job.state in FINAL_STATES:
                return job
            if job.state == JobState.QUEUED:
                job.state = JobState.CANCELLED
            job.cancel_requested = True
            self._save(c, job, row["token"], row["lease_until"])
            return job

    def retry(self, job_id: str, actor: str) -> QueueJob:
        with self._tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            job = self._job(row)
            if job.state != JobState.FAILED:
                raise ValueError(f"job {job_id} is {job.state.value}, only failed jobs can be retried")
            job.state = JobState.QUEUED
            job.cancel_requested = False
            job.machine = None
            self._save(c, job, None, None)
            self._audit(c, actor, "retry", job_id)
            return job

    def get(self, job_id: str) -> Optional[QueueJob]:
        with self._tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self._job(row) if row else None

    def list(self, states: Optional[set[JobState]] = None, limit: int = 100) -> list[QueueJob]:
        with self._tx() as c:
            sql, args = "SELECT * FROM jobs", []
            if states:
                sql += f" WHERE state IN ({','.join('?' * len(states))})"
                args += [JobState(s).value for s in states]
            args.append(limit)
            return [self._job(r) for r in c.execute(sql + " ORDER BY submitted_at, seq LIMIT ?", args)]

    def stats(self) -> QueueStats:
        with self._tx() as c:
            by_state = {r["state"]: r["n"] for r in c.execute("SELECT state, COUNT(*) n FROM jobs GROUP BY state")}
            oldest = c.execute(
                "SELECT MIN(submitted_at) m FROM jobs WHERE state=?", (JobState.QUEUED.value,)
            ).fetchone()["m"]
            return QueueStats(by_state, None if oldest is None else self.now() - oldest)

    def _meta(self, c: sqlite3.Connection, key: str) -> Optional[str]:
        r = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

    def _set_meta(self, c: sqlite3.Connection, key: str, value: Optional[str]) -> None:
        if value is None:
            c.execute("DELETE FROM meta WHERE key=?", (key,))
        else:
            c.execute("INSERT INTO meta (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def _coord(self, c: sqlite3.Connection) -> Optional[dict]:
        owner, lease = self._meta(c, "coordinator_owner"), self._meta(c, "coordinator_lease")
        if owner is None or lease is None or float(lease) <= self.now():
            return None
        return {"owner": owner, "lease_until": float(lease)}

    def _take(self, c: sqlite3.Connection, owner: str, lease_s: float) -> None:
        self._set_meta(c, "coordinator_owner", owner)
        self._set_meta(c, "coordinator_lease", str(self.now() + lease_s))

    def acquire_coordinator(self, owner: str, lease_s: float) -> bool:
        with self._tx() as c:
            cur = self._coord(c)
            if cur is not None and cur["owner"] != owner:
                return False
            self._take(c, owner, lease_s)
            return True

    def renew_coordinator(self, owner: str, lease_s: float) -> bool:
        with self._tx() as c:
            cur = self._coord(c)
            if cur is None or cur["owner"] != owner:
                return False
            self._take(c, owner, lease_s)
            return True

    def release_coordinator(self, owner: str) -> None:
        with self._tx() as c:
            cur = self._coord(c)
            if cur is not None and cur["owner"] == owner:
                self._set_meta(c, "coordinator_owner", None)
                self._set_meta(c, "coordinator_lease", None)

    def coordinator(self) -> Optional[dict]:
        with self._tx() as c:
            return self._coord(c)

    def set_paused(self, paused: bool, actor: str) -> None:
        with self._tx() as c:
            self._set_meta(c, "paused", "1" if paused else "0")
            self._audit(c, actor, "set_paused", None, str(bool(paused)).lower())

    def paused(self) -> bool:
        with self._tx() as c:
            return self._meta(c, "paused") == "1"


def create(config: Any) -> LocalBackend:
    return LocalBackend(Path(config.state_dir) / "queue" / "queue.db")
