"""Operational monitoring utilities for XRayMind case workflows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .export import build_case_export_rows
from .store import DEFAULT_DB_PATH, SQLiteStore, utc_now_iso


def _rate(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else 0.0


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for label in row.get("top_finding_labels") or []:
            counts[str(label)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("status") or "unknown") for row in rows))


def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("latest_decision") or "unreviewed") for row in rows))


def build_monitoring_snapshot(
    *,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 10_000,
    drift_baseline: dict[str, Any] | None = None,
    drift_threshold: float = 0.15,
) -> dict[str, Any]:
    """Build a compact snapshot of case quality and model workflow health."""

    rows = build_case_export_rows(limit=limit, store=store, db_path=db_path)
    total = len(rows)
    low_confidence_count = sum(1 for row in rows if row.get("low_confidence") is True)
    reviewed_count = sum(1 for row in rows if (row.get("review_count") or 0) > 0)
    disagreement_count = sum(1 for row in rows if row.get("latest_decision") == "disagree")
    flagged_count = sum(1 for row in rows if row.get("status") == "flagged")
    deferred_count = sum(1 for row in rows if row.get("status") == "deferred")
    urgent_count = sum(1 for row in rows if row.get("priority") == "urgent")

    labels = _label_counts(rows)
    snapshot: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "total_cases": total,
        "reviewed_cases": reviewed_count,
        "rates": {
            "review_rate": _rate(reviewed_count, total),
            "low_confidence_rate": _rate(low_confidence_count, total),
            "disagreement_rate": _rate(disagreement_count, max(reviewed_count, 1)),
            "flagged_rate": _rate(flagged_count, total),
            "deferred_rate": _rate(deferred_count, total),
            "urgent_rate": _rate(urgent_count, total),
        },
        "counts": {
            "low_confidence": low_confidence_count,
            "disagreements": disagreement_count,
            "flagged": flagged_count,
            "deferred": deferred_count,
            "urgent": urgent_count,
            "status": _status_counts(rows),
            "latest_decision": _decision_counts(rows),
            "top_finding_labels": labels,
        },
        "alerts": [],
        "disclaimer": "Research workflow monitoring only. Not for clinical triage.",
    }

    alerts: list[dict[str, Any]] = []
    if total and snapshot["rates"]["low_confidence_rate"] >= 0.25:
        alerts.append({"level": "warning", "type": "low_confidence", "message": "Low-confidence cases exceed 25% of the export window."})
    if reviewed_count >= 5 and snapshot["rates"]["disagreement_rate"] >= 0.20:
        alerts.append({"level": "warning", "type": "review_disagreement", "message": "Reviewer disagreement exceeds 20% of reviewed cases."})
    if total and snapshot["rates"]["flagged_rate"] >= 0.10:
        alerts.append({"level": "warning", "type": "flagged_cases", "message": "Flagged cases exceed 10% of the export window."})

    if drift_baseline:
        baseline_rates = drift_baseline.get("rates", {}) or {}
        rate_deltas = {
            key: snapshot["rates"].get(key, 0.0) - float(baseline_rates.get(key, 0.0))
            for key in snapshot["rates"]
        }
        snapshot["drift"] = {
            "baseline_created_at": drift_baseline.get("created_at"),
            "rate_deltas": rate_deltas,
            "threshold": drift_threshold,
        }
        for key, delta in rate_deltas.items():
            if abs(delta) >= drift_threshold:
                alerts.append(
                    {
                        "level": "warning",
                        "type": "rate_drift",
                        "metric": key,
                        "delta": delta,
                        "message": f"{key} changed by {delta:.3f} versus baseline.",
                    }
                )

    snapshot["alerts"] = alerts
    return snapshot


def save_monitoring_snapshot(
    out: str | Path = "outputs/monitoring/snapshot.json",
    *,
    baseline: str | Path | None = None,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 10_000,
    drift_threshold: float = 0.15,
) -> dict[str, Any]:
    """Write a monitoring snapshot to disk and return it."""

    baseline_payload: dict[str, Any] | None = None
    if baseline:
        baseline_path = Path(baseline)
        if baseline_path.exists():
            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))

    snapshot = build_monitoring_snapshot(
        store=store,
        db_path=db_path,
        limit=limit,
        drift_baseline=baseline_payload,
        drift_threshold=drift_threshold,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return snapshot
