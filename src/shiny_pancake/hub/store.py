"""Durable hub state in SQLite: a job queue plus the trace events of every run.

SQLite in WAL mode lets the API process and any number of ``shiny-pancake worker`` processes
share one database file (e.g. a Docker volume). Jobs are claimed atomically with
``BEGIN IMMEDIATE``, so two workers can never run the same job; stale ``running`` jobs left
behind by a crashed worker are re-queued on startup.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..tracing import TraceEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    result      TEXT,
    error       TEXT,
    worker      TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT NOT NULL,
    seq     INTEGER NOT NULL,
    ts      REAL NOT NULL,
    agent   TEXT NOT NULL,
    kind    TEXT NOT NULL,
    data    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_job ON events(job_id, id);
"""

JOB_STATUSES = ("queued", "running", "succeeded", "failed")


@dataclass
class Job:
    id: str
    kind: str
    status: str
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    worker: str | None = None
    attempts: int = 0
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def done(self) -> bool:
        return self.status in ("succeeded", "failed")

    def to_dict(self, *, include_result: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_result:
            data.pop("result")
        end = self.finished_at or time.time()
        data["duration_s"] = round(end - self.started_at, 3) if self.started_at else None
        return data


class HubStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ jobs
    def create_job(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        job_id: str | None = None,
        status: str = "queued",
        worker: str | None = None,
    ) -> Job:
        job = Job(
            job_id or uuid.uuid4().hex[:12], kind, status, payload, worker=worker, created_at=time.time()
        )
        if status == "running":
            job.started_at = job.created_at
            job.attempts = 1
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, status, payload, worker, attempts, created_at, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id,
                    kind,
                    status,
                    json.dumps(payload, ensure_ascii=False),
                    worker,
                    job.attempts,
                    job.created_at,
                    job.started_at,
                ),
            )
        return job

    def claim_next(self, worker: str, kinds: tuple[str, ...] | None = None) -> Job | None:
        """Atomically move the oldest queued job to ``running`` and return it."""
        where = "status = 'queued'"
        params: list[Any] = []
        if kinds:
            where += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    f"SELECT id FROM jobs WHERE {where} ORDER BY created_at LIMIT 1", params
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None
                self._conn.execute(
                    "UPDATE jobs SET status='running', worker=?, started_at=?, attempts=attempts+1 "
                    "WHERE id=?",
                    (worker, time.time(), row["id"]),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return self.get_job(row["id"])

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='succeeded', result=?, finished_at=? WHERE id=?",
                (json.dumps(result, ensure_ascii=False, default=str), time.time(), job_id),
            )

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?",
                (error[:4000], time.time(), job_id),
            )

    def requeue_stale(self, older_than_s: float = 3600) -> int:
        """Crash recovery: jobs stuck in ``running`` longer than the limit go back to the queue."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='queued', worker=NULL WHERE status='running' AND started_at < ?",
                (time.time() - older_than_s,),
            )
            return cur.rowcount

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, *, limit: int = 50, status: str | None = None, kind: str | None = None) -> list[Job]:
        sql, params = "SELECT * FROM jobs WHERE 1=1", []
        if status:
            sql += " AND status=?"
            params.append(status)
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_job(r) for r in rows]

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        counts = {s: 0 for s in JOB_STATUSES}
        counts.update({r["status"]: r["n"] for r in rows})
        return counts

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            payload=json.loads(row["payload"]),
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            worker=row["worker"],
            attempts=row["attempts"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    # ------------------------------------------------------------------ events
    def append_event(self, event: TraceEvent) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (job_id, seq, ts, agent, kind, data) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    event.seq,
                    event.ts,
                    event.agent,
                    event.kind,
                    json.dumps(event.data, ensure_ascii=False, default=str),
                ),
            )

    def events(self, job_id: str, *, after_id: int = 0, limit: int = 2000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT ?", (job_id, after_id, limit)
            ).fetchall()
        return [
            {
                "id": r["id"],
                "run_id": r["job_id"],
                "seq": r["seq"],
                "ts": r["ts"],
                "agent": r["agent"],
                "kind": r["kind"],
                "data": json.loads(r["data"]),
            }
            for r in rows
        ]


class StoreSink:
    """Tracer sink that persists events, which is what makes live streaming work across processes."""

    def __init__(self, store: HubStore):
        self.store = store

    def emit(self, event: TraceEvent) -> None:
        self.store.append_event(event)
