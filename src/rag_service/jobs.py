"""Tiny job table so long-running index/embedding runs can be triggered over HTTP and polled."""

import logging
import threading
from typing import Callable

from psycopg.types.json import Jsonb

from .db import get_conn

log = logging.getLogger(__name__)

# One run per stage at a time, otherwise two workers would grab the same pending rows.
_locks = {"embedding": threading.Lock(), "graph_index": threading.Lock()}


def create_job(kind: str, params: dict) -> str:
    with get_conn() as conn:
        return conn.execute(
            "INSERT INTO jobs (kind, params) VALUES (%s, %s) RETURNING id::text AS id", (kind, Jsonb(params))
        ).fetchone()["id"]


def get_job(job_id: str) -> dict | None:
    with get_conn() as conn:
        return conn.execute(
            """SELECT id::text AS id, kind, status, params, progress, error, created_at, finished_at
               FROM jobs WHERE id = %s""",
            (job_id,),
        ).fetchone()


def list_jobs(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT id::text AS id, kind, status, params, progress, error, created_at, finished_at
               FROM jobs ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        ).fetchall()


def _set(job_id: str, **fields) -> None:
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    if "progress" in fields:
        fields["progress"] = Jsonb(fields["progress"])
    with get_conn() as conn:
        conn.execute(f"UPDATE jobs SET {sets} WHERE id = %(id)s", {**fields, "id": job_id})


def is_busy(kind: str) -> bool:
    return _locks[kind].locked()


def run_job(job_id: str, kind: str, fn: Callable[..., dict], **kwargs) -> None:
    lock = _locks[kind]
    if not lock.acquire(blocking=False):
        _set(job_id, status="rejected", error=f"another {kind} job is already running")
        return
    try:
        _set(job_id, status="running")
        result = fn(progress=lambda p: _set(job_id, progress=dict(p)), **kwargs)
        _set(job_id, status="succeeded", progress=result)
        with get_conn() as conn:
            conn.execute("UPDATE jobs SET finished_at = now() WHERE id = %s", (job_id,))
    except Exception as exc:
        log.exception("job %s failed", job_id)
        _set(job_id, status="failed", error=str(exc)[:4000])
        with get_conn() as conn:
            conn.execute("UPDATE jobs SET finished_at = now() WHERE id = %s", (job_id,))
    finally:
        lock.release()
