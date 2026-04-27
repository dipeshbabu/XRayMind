"""Case export utilities for XRayMind.

The export layer is intentionally lightweight: it reads the local SQLite
workflow database and creates analysis-ready JSONL/CSV files plus a manifest
with checksums. This is meant for audits, human-review studies, and offline
quality monitoring, not clinical reporting.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .cases import get_latest_prediction, list_cases, list_reviews
from .store import DEFAULT_DB_PATH, SQLiteStore, dumps_json, utc_now_iso


def _coerce_store(store: SQLiteStore | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> SQLiteStore:
    return store if store is not None else SQLiteStore(db_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _top_finding_labels(prediction: dict[str, Any] | None) -> list[str]:
    if not prediction:
        return []
    payload = prediction.get("prediction_json", {}) or {}
    findings = payload.get("top_findings", []) or []
    labels: list[str] = []
    for item in findings:
        if isinstance(item, dict) and item.get("label"):
            labels.append(str(item["label"]))
    return labels


def build_case_export_rows(
    *,
    status: str | None = None,
    priority: str | None = None,
    limit: int = 10_000,
    offset: int = 0,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return one denormalized row per case for export and monitoring."""

    store = _coerce_store(store, db_path)
    cases = list_cases(status=status, priority=priority, limit=limit, offset=offset, store=store)
    rows: list[dict[str, Any]] = []
    for case in cases:
        prediction = get_latest_prediction(case["id"], store=store)
        reviews = list_reviews(case["id"], store=store)
        latest_review = reviews[-1] if reviews else None
        prediction_payload = prediction.get("prediction_json", {}) if prediction else {}
        uncertainty = prediction_payload.get("uncertainty", {}) if isinstance(prediction_payload, dict) else {}
        rows.append(
            {
                "case_id": case["id"],
                "image_path": case.get("image_path"),
                "image_id": case.get("image_id"),
                "source_filename": case.get("source_filename"),
                "model_name": case.get("model_name"),
                "status": case.get("status"),
                "priority": case.get("priority"),
                "assigned_to": case.get("assigned_to"),
                "due_at": case.get("due_at"),
                "needs_second_reader": bool(case.get("needs_second_reader", False)),
                "tags": case.get("tags", []),
                "created_at": case.get("created_at"),
                "updated_at": case.get("updated_at"),
                "prediction_id": prediction.get("id") if prediction else None,
                "prediction_created_at": prediction.get("created_at") if prediction else None,
                "threshold": prediction.get("threshold") if prediction else None,
                "top_k": prediction.get("top_k") if prediction else None,
                "max_probability": prediction.get("max_probability") if prediction else None,
                "low_confidence": bool(prediction.get("low_confidence")) if prediction else None,
                "top_finding_labels": _top_finding_labels(prediction),
                "uncertainty": uncertainty or {},
                "review_count": len(reviews),
                "latest_reviewer": latest_review.get("reviewer") if latest_review else None,
                "latest_decision": latest_review.get("decision") if latest_review else None,
                "latest_review_notes": latest_review.get("notes") if latest_review else None,
                "latest_final_labels": latest_review.get("final_labels") if latest_review else {},
            }
        )
    return rows


def export_cases(
    out_dir: str | Path = "outputs/exports",
    *,
    status: str | None = None,
    priority: str | None = None,
    limit: int = 10_000,
    offset: int = 0,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Export case workflow data to JSONL and CSV with a manifest."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    store = _coerce_store(store, db_path)
    rows = build_case_export_rows(
        status=status,
        priority=priority,
        limit=limit,
        offset=offset,
        store=store,
    )

    jsonl_path = out_path / "cases_export.jsonl"
    csv_path = out_path / "cases_export.csv"
    manifest_path = out_path / "manifest.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    csv_columns = [
        "case_id",
        "image_id",
        "source_filename",
        "model_name",
        "status",
        "priority",
        "assigned_to",
        "due_at",
        "needs_second_reader",
        "created_at",
        "updated_at",
        "prediction_id",
        "prediction_created_at",
        "threshold",
        "top_k",
        "max_probability",
        "low_confidence",
        "top_finding_labels",
        "review_count",
        "latest_reviewer",
        "latest_decision",
        "latest_final_labels",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["top_finding_labels"] = ";".join(row.get("top_finding_labels") or [])
            flat["latest_final_labels"] = dumps_json(row.get("latest_final_labels") or {})
            writer.writerow({key: flat.get(key) for key in csv_columns})

    manifest = {
        "created_at": utc_now_iso(),
        "row_count": len(rows),
        "filters": {"status": status, "priority": priority, "limit": limit, "offset": offset},
        "files": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
        },
        "sha256": {
            "jsonl": _sha256(jsonl_path),
            "csv": _sha256(csv_path),
        },
        "disclaimer": "Research workflow export only. Not for clinical diagnosis.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["files"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
