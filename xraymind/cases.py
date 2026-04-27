"""Case workflow helpers for XRayMind."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from .inference import predict_image
from .store import DEFAULT_DB_PATH, SQLiteStore, dumps_json, loads_json, utc_now_iso

VALID_CASE_STATUSES = {"pending", "reviewed", "deferred", "flagged", "archived"}
VALID_PRIORITIES = {"routine", "elevated", "urgent"}
VALID_REVIEW_DECISIONS = {"agree", "disagree", "uncertain", "defer", "flag"}


def _coerce_store(store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> SQLiteStore:
    return store if store is not None else SQLiteStore(db_path)


def _validate_choice(value: str, valid: set[str], field: str) -> str:
    normalized = value.strip().lower()
    if normalized not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"Invalid {field}: {value!r}. Expected one of: {choices}")
    return normalized


def _hydrate_case(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    hydrated = dict(row)
    hydrated["tags"] = loads_json(hydrated.get("tags"), default=[])
    hydrated["patient_context"] = loads_json(hydrated.get("patient_context"), default={})
    return hydrated


def _hydrate_prediction(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    hydrated = dict(row)
    hydrated["prediction_json"] = loads_json(hydrated.get("prediction_json"), default={})
    hydrated["low_confidence"] = bool(hydrated.get("low_confidence"))
    return hydrated


def _hydrate_review(row: dict[str, Any]) -> dict[str, Any]:
    hydrated = dict(row)
    hydrated["final_labels"] = loads_json(hydrated.get("final_labels"), default={})
    return hydrated


def create_case(
    image_path: str | Path,
    *,
    image_id: str | None = None,
    source_filename: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    status: str = "pending",
    priority: str = "routine",
    patient_context: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Create a case record without running inference."""

    image_path = str(image_path)
    status = _validate_choice(status, VALID_CASE_STATUSES, "status")
    priority = _validate_choice(priority, VALID_PRIORITIES, "priority")
    now = utc_now_iso()
    store = _coerce_store(store, db_path)
    case_id = store.execute(
        """
        INSERT INTO cases(
            image_path, image_id, source_filename, model_name, status, priority,
            patient_context, tags, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_path,
            image_id,
            source_filename or Path(image_path).name,
            model_name,
            status,
            priority,
            dumps_json(patient_context or {}),
            dumps_json(tags or []),
            now,
            now,
        ),
    )
    log_event(case_id, "case.created", {"image_path": image_path, "status": status}, store=store)
    case = get_case(case_id, store=store)
    assert case is not None
    return case


def save_prediction_for_case(
    case_id: int,
    prediction: dict[str, Any],
    *,
    model_name: str,
    threshold: float,
    top_k: int,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Persist a prediction payload for an existing case."""

    store = _coerce_store(store, db_path)
    uncertainty = prediction.get("uncertainty", {}) or {}
    prediction_id = store.execute(
        """
        INSERT INTO predictions(
            case_id, model_name, threshold, top_k, prediction_json,
            max_probability, low_confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(case_id),
            model_name,
            float(threshold),
            int(top_k),
            dumps_json(prediction),
            float(uncertainty.get("max_probability", 0.0)),
            int(bool(uncertainty.get("low_confidence", False))),
            utc_now_iso(),
        ),
    )
    log_event(case_id, "prediction.created", {"prediction_id": prediction_id, "model_name": model_name}, store=store)
    saved = get_prediction(prediction_id, store=store)
    assert saved is not None
    return saved


def create_case_with_prediction(
    image_path: str | Path,
    *,
    image_id: str | None = None,
    source_filename: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = 0.5,
    priority: str = "routine",
    patient_context: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Create a case, run inference, and persist the prediction."""

    store = _coerce_store(store, db_path)
    case = create_case(
        image_path,
        image_id=image_id,
        source_filename=source_filename,
        model_name=model_name,
        priority=priority,
        patient_context=patient_context,
        tags=tags,
        store=store,
    )
    prediction = predict_image(image_path, model_name=model_name, top_k=top_k, threshold=threshold)
    saved_prediction = save_prediction_for_case(
        case["id"],
        prediction,
        model_name=model_name,
        threshold=threshold,
        top_k=top_k,
        store=store,
    )
    return {"case": case, "prediction": saved_prediction}


def get_case(case_id: int, *, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Return one case by id."""

    store = _coerce_store(store, db_path)
    return _hydrate_case(store.fetch_one("SELECT * FROM cases WHERE id = ?", (int(case_id),)))


def get_prediction(prediction_id: int, *, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Return one prediction by id."""

    store = _coerce_store(store, db_path)
    return _hydrate_prediction(store.fetch_one("SELECT * FROM predictions WHERE id = ?", (int(prediction_id),)))


def get_latest_prediction(case_id: int, *, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Return the most recent prediction for a case."""

    store = _coerce_store(store, db_path)
    row = store.fetch_one(
        "SELECT * FROM predictions WHERE case_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
        (int(case_id),),
    )
    return _hydrate_prediction(row)


def list_reviews(case_id: int, *, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Return reviews for a case in creation order."""

    store = _coerce_store(store, db_path)
    rows = store.fetch_all(
        "SELECT * FROM reviews WHERE case_id = ? ORDER BY created_at ASC, id ASC",
        (int(case_id),),
    )
    return [_hydrate_review(row) for row in rows]


def get_case_detail(case_id: int, *, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Return case, latest prediction, and reviews."""

    store = _coerce_store(store, db_path)
    case = get_case(case_id, store=store)
    if case is None:
        raise KeyError(f"Case {case_id} not found")
    return {
        "case": case,
        "latest_prediction": get_latest_prediction(case_id, store=store),
        "reviews": list_reviews(case_id, store=store),
    }


def list_cases(
    *,
    status: str | None = None,
    priority: str | None = None,
    limit: int = 50,
    offset: int = 0,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """List cases with optional status and priority filtering."""

    store = _coerce_store(store, db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(_validate_choice(status, VALID_CASE_STATUSES, "status"))
    if priority:
        clauses.append("priority = ?")
        params.append(_validate_choice(priority, VALID_PRIORITIES, "priority"))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([int(limit), int(offset)])
    rows = store.fetch_all(
        f"SELECT * FROM cases {where} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
        tuple(params),
    )
    return [_hydrate_case(row) for row in rows if row is not None]


def update_case_status(
    case_id: int,
    status: str,
    *,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Update case status and return the case."""

    status = _validate_choice(status, VALID_CASE_STATUSES, "status")
    store = _coerce_store(store, db_path)
    store.execute(
        "UPDATE cases SET status = ?, updated_at = ? WHERE id = ?",
        (status, utc_now_iso(), int(case_id)),
    )
    log_event(case_id, "case.status_updated", {"status": status}, store=store)
    case = get_case(case_id, store=store)
    if case is None:
        raise KeyError(f"Case {case_id} not found")
    return case


def add_review(
    case_id: int,
    *,
    decision: str,
    reviewer: str | None = None,
    notes: str | None = None,
    final_labels: dict[str, Any] | None = None,
    next_status: str | None = None,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Add a human review and update case status."""

    decision = _validate_choice(decision, VALID_REVIEW_DECISIONS, "decision")
    store = _coerce_store(store, db_path)
    now = utc_now_iso()
    review_id = store.execute(
        """
        INSERT INTO reviews(case_id, reviewer, decision, notes, final_labels, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (int(case_id), reviewer, decision, notes, dumps_json(final_labels or {}), now),
    )
    status = next_status
    if status is None:
        status = "deferred" if decision == "defer" else "flagged" if decision == "flag" else "reviewed"
    update_case_status(case_id, status, store=store)
    log_event(case_id, "review.created", {"review_id": review_id, "decision": decision}, store=store)
    row = store.fetch_one("SELECT * FROM reviews WHERE id = ?", (review_id,))
    assert row is not None
    return _hydrate_review(row)


def log_event(
    case_id: int | None,
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Write an audit event to the workflow database."""

    store = _coerce_store(store, db_path)
    return store.execute(
        "INSERT INTO audit_events(case_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
        (case_id, event_type, dumps_json(payload or {}), utc_now_iso()),
    )
