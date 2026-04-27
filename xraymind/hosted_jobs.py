"""Hosted asynchronous job queue helpers for XRayMind.

This module keeps the first hosted-job implementation deliberately small:
SQLite backed, tenant aware, deterministic in tests, and safe for CI because
nothing runs in a background thread unless a caller explicitly processes a job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cases import create_case_with_prediction, get_case_detail, log_event
from .config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from .store import DEFAULT_DB_PATH, SQLiteStore, dumps_json, loads_json, utc_now_iso

VALID_JOB_STATUSES = {"queued", "running", "succeeded", "failed", "cancelled"}
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}


def _coerce_store(store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> SQLiteStore:
    return store if store is not None else SQLiteStore(db_path)


def _validate_choice(value: str, valid: set[str], field: str) -> str:
    normalized = value.strip().lower()
    if normalized not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"Invalid {field}: {value!r}. Expected one of: {choices}")
    return normalized


def _hydrate_job(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    hydrated = dict(row)
    hydrated["payload"] = loads_json(hydrated.get("payload"), default={})
    hydrated["result"] = loads_json(hydrated.get("result"), default=None)
    hydrated["error"] = loads_json(hydrated.get("error"), default=None)
    return hydrated


def enqueue_case_prediction_job(
    image_path: str | Path,
    *,
    tenant_id: str = "default",
    image_id: str | None = None,
    source_filename: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = 0.5,
    priority: str = "routine",
    tags: list[str] | None = None,
    assigned_to: str | None = None,
    due_at: str | None = None,
    needs_second_reader: bool = False,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Create a queued hosted job for an uploaded image."""

    store = _coerce_store(store, db_path)
    now = utc_now_iso()
    payload = {
        "image_path": str(image_path),
        "image_id": image_id,
        "source_filename": source_filename or Path(image_path).name,
        "model_name": model_name,
        "top_k": int(top_k),
        "threshold": float(threshold),
        "priority": priority,
        "tags": tags or [],
        "assigned_to": assigned_to,
        "due_at": due_at,
        "needs_second_reader": bool(needs_second_reader),
    }
    job_id = store.execute(
        """
        INSERT INTO hosted_jobs(
            tenant_id, job_type, status, payload, result, error,
            attempts, created_at, updated_at, started_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tenant_id,
            "case_prediction",
            "queued",
            dumps_json(payload),
            None,
            None,
            0,
            now,
            now,
            None,
            None,
        ),
    )
    log_event(None, "job.queued", {"job_id": job_id, "tenant_id": tenant_id, "job_type": "case_prediction"}, store=store)
    job = get_job(job_id, store=store)
    assert job is not None
    return job


def get_job(job_id: int, *, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Return one hosted job."""

    store = _coerce_store(store, db_path)
    return _hydrate_job(store.fetch_one("SELECT * FROM hosted_jobs WHERE id = ?", (int(job_id),)))


def list_jobs(
    *,
    tenant_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """List hosted jobs with optional tenant/status filtering."""

    store = _coerce_store(store, db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if tenant_id:
        clauses.append("tenant_id = ?")
        params.append(tenant_id)
    if status:
        clauses.append("status = ?")
        params.append(_validate_choice(status, VALID_JOB_STATUSES, "status"))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([int(limit), int(offset)])
    rows = store.fetch_all(
        f"SELECT * FROM hosted_jobs {where} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
        tuple(params),
    )
    return [job for row in rows if (job := _hydrate_job(row)) is not None]


def _set_job_status(
    job_id: int,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    store: SQLiteStore,
) -> dict[str, Any]:
    status = _validate_choice(status, VALID_JOB_STATUSES, "status")
    now = utc_now_iso()
    updates = ["status = ?", "updated_at = ?"]
    params: list[Any] = [status, now]
    if status == "running":
        updates.append("started_at = COALESCE(started_at, ?)")
        updates.append("attempts = attempts + 1")
        params.append(now)
    if status in TERMINAL_JOB_STATUSES:
        updates.append("completed_at = ?")
        params.append(now)
    if result is not None:
        updates.append("result = ?")
        params.append(dumps_json(result))
    if error is not None:
        updates.append("error = ?")
        params.append(dumps_json(error))
    params.append(int(job_id))
    store.execute(f"UPDATE hosted_jobs SET {', '.join(updates)} WHERE id = ?", tuple(params))
    job = get_job(job_id, store=store)
    if job is None:
        raise KeyError(f"Job {job_id} not found")
    return job


def process_next_job(*, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Process the oldest queued job synchronously and return the final job."""

    store = _coerce_store(store, db_path)
    row = store.fetch_one("SELECT * FROM hosted_jobs WHERE status = 'queued' ORDER BY created_at ASC, id ASC LIMIT 1")
    job = _hydrate_job(row)
    if job is None:
        return None
    _set_job_status(job["id"], "running", store=store)
    try:
        result = _run_job(job, store=store)
    except Exception as exc:
        error = {"type": exc.__class__.__name__, "message": str(exc)}
        failed = _set_job_status(job["id"], "failed", error=error, store=store)
        log_event(None, "job.failed", {"job_id": job["id"], "error": error}, store=store)
        return failed
    completed = _set_job_status(job["id"], "succeeded", result=result, store=store)
    log_event(result.get("case_id"), "job.succeeded", {"job_id": job["id"], **result}, store=store)
    return completed


def cancel_job(job_id: int, *, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Cancel a queued job before a worker starts it."""

    store = _coerce_store(store, db_path)
    job = get_job(job_id, store=store)
    if job is None:
        raise KeyError(f"Job {job_id} not found")
    if job["status"] != "queued":
        raise ValueError("Only queued jobs can be cancelled")
    cancelled = _set_job_status(job_id, "cancelled", store=store)
    log_event(None, "job.cancelled", {"job_id": job_id}, store=store)
    return cancelled


def _run_job(job: dict[str, Any], *, store: SQLiteStore) -> dict[str, Any]:
    if job["job_type"] != "case_prediction":
        raise ValueError(f"Unsupported job type: {job['job_type']}")
    payload = job["payload"]
    created = create_case_with_prediction(
        payload["image_path"],
        image_id=payload.get("image_id"),
        source_filename=payload.get("source_filename"),
        model_name=payload.get("model_name", DEFAULT_MODEL_NAME),
        top_k=int(payload.get("top_k", DEFAULT_TOP_K)),
        threshold=float(payload.get("threshold", 0.5)),
        priority=payload.get("priority", "routine"),
        tags=payload.get("tags") or [],
        assigned_to=payload.get("assigned_to"),
        due_at=payload.get("due_at"),
        needs_second_reader=bool(payload.get("needs_second_reader", False)),
        store=store,
    )
    case_id = int(created["case"]["id"])
    prediction_id = int(created["prediction"]["id"])
    return {"case_id": case_id, "prediction_id": prediction_id, "case_detail": get_case_detail(case_id, store=store)}
