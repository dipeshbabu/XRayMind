"""Dashboard summaries for the local XRayMind case workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .store import DEFAULT_DB_PATH, SQLiteStore


def _coerce_store(store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> SQLiteStore:
    return store if store is not None else SQLiteStore(db_path)


def _count(store: SQLiteStore, query: str, params: tuple[Any, ...] = ()) -> int:
    row = store.fetch_one(query, params) or {}
    return int(row.get("count", 0) or 0)


def dashboard_summary(*, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Return aggregate counts for the review workflow."""

    store = _coerce_store(store, db_path)
    status_rows = store.fetch_all("SELECT status, COUNT(*) AS count FROM cases GROUP BY status")
    priority_rows = store.fetch_all("SELECT priority, COUNT(*) AS count FROM cases GROUP BY priority")
    review_rows = store.fetch_all("SELECT decision, COUNT(*) AS count FROM reviews GROUP BY decision")
    label_rows = store.fetch_all(
        """
        SELECT json_extract(value.value, '$.label') AS label, COUNT(*) AS count
        FROM predictions, json_each(predictions.prediction_json, '$.top_findings') AS value
        WHERE json_extract(value.value, '$.label') IS NOT NULL
        GROUP BY label
        ORDER BY count DESC, label ASC
        LIMIT 20
        """
    )
    total_cases = _count(store, "SELECT COUNT(*) AS count FROM cases")
    reviewed_cases = _count(store, "SELECT COUNT(*) AS count FROM cases WHERE status = 'reviewed'")
    deferred_cases = _count(store, "SELECT COUNT(*) AS count FROM cases WHERE status = 'deferred'")
    flagged_cases = _count(store, "SELECT COUNT(*) AS count FROM cases WHERE status = 'flagged'")
    low_confidence_cases = _count(
        store,
        """
        SELECT COUNT(DISTINCT case_id) AS count
        FROM predictions
        WHERE low_confidence = 1
        """,
    )
    disagreement_reviews = _count(store, "SELECT COUNT(*) AS count FROM reviews WHERE decision = 'disagree'")
    total_reviews = _count(store, "SELECT COUNT(*) AS count FROM reviews")
    return {
        "total_cases": total_cases,
        "reviewed_cases": reviewed_cases,
        "pending_cases": _count(store, "SELECT COUNT(*) AS count FROM cases WHERE status = 'pending'"),
        "deferred_cases": deferred_cases,
        "flagged_cases": flagged_cases,
        "low_confidence_cases": low_confidence_cases,
        "total_reviews": total_reviews,
        "reviewer_disagreement_rate": round(disagreement_reviews / total_reviews, 6) if total_reviews else 0.0,
        "status_counts": {row["status"]: int(row["count"]) for row in status_rows},
        "priority_counts": {row["priority"]: int(row["count"]) for row in priority_rows},
        "review_decision_counts": {row["decision"]: int(row["count"]) for row in review_rows},
        "top_alert_labels": [
            {"label": row["label"], "count": int(row["count"])} for row in label_rows if row.get("label")
        ],
    }


def cases_requiring_attention(*, store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH, limit: int = 25) -> list[dict[str, Any]]:
    """Return cases that should be prioritized for human review."""

    store = _coerce_store(store, db_path)
    return store.fetch_all(
        """
        SELECT
            cases.*,
            predictions.max_probability,
            predictions.low_confidence,
            predictions.created_at AS prediction_created_at
        FROM cases
        LEFT JOIN predictions ON predictions.id = (
            SELECT id FROM predictions p2
            WHERE p2.case_id = cases.id
            ORDER BY p2.created_at DESC, p2.id DESC
            LIMIT 1
        )
        WHERE cases.status IN ('pending', 'deferred', 'flagged')
           OR predictions.low_confidence = 1
        ORDER BY
            CASE cases.priority WHEN 'urgent' THEN 0 WHEN 'elevated' THEN 1 ELSE 2 END,
            CASE cases.status WHEN 'flagged' THEN 0 WHEN 'deferred' THEN 1 WHEN 'pending' THEN 2 ELSE 3 END,
            cases.created_at DESC
        LIMIT ?
        """,
        (int(limit),),
    )
