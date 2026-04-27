"""Operational monitoring utilities for XRayMind case workflows."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

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


def _review_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    reviewed_count = sum(1 for row in rows if (row.get("review_count") or 0) > 0)
    low_confidence_count = sum(1 for row in rows if row.get("low_confidence") is True)
    disagreement_count = sum(1 for row in rows if row.get("latest_decision") == "disagree")
    flagged_count = sum(1 for row in rows if row.get("status") == "flagged")
    deferred_count = sum(1 for row in rows if row.get("status") == "deferred")
    urgent_count = sum(1 for row in rows if row.get("priority") == "urgent")
    return {
        "n": total,
        "reviewed": reviewed_count,
        "low_confidence": low_confidence_count,
        "disagreements": disagreement_count,
        "flagged": flagged_count,
        "deferred": deferred_count,
        "urgent": urgent_count,
        "review_rate": _rate(reviewed_count, total),
        "low_confidence_rate": _rate(low_confidence_count, total),
        "disagreement_rate": _rate(disagreement_count, max(reviewed_count, 1)),
        "flagged_rate": _rate(flagged_count, total),
        "deferred_rate": _rate(deferred_count, total),
        "urgent_rate": _rate(urgent_count, total),
    }


def _row_group_values(row: dict[str, Any], field: str) -> Iterable[str]:
    value = row.get(field)
    if field == "tags":
        tags = value or []
        if not tags:
            yield "untagged"
        for tag in tags:
            yield str(tag)
        return
    if isinstance(value, list):
        if not value:
            yield "none"
        for item in value:
            yield str(item)
        return
    yield str(value if value not in (None, "") else "unknown")


def build_subgroup_audit(
    rows: list[dict[str, Any]],
    *,
    fields: list[str] | None = None,
    min_cases: int = 1,
) -> list[dict[str, Any]]:
    """Aggregate workflow quality metrics by subgroup fields.

    This is not a fairness certification. It is a lightweight operational audit
    to surface which review queues, tags, models, or statuses need attention.
    """

    selected_fields = fields or ["priority", "status", "model_name", "tags"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for field in selected_fields:
            for value in _row_group_values(row, field):
                groups[(field, value)].append(row)

    audit_rows: list[dict[str, Any]] = []
    for (field, value), group_rows in groups.items():
        if len(group_rows) < min_cases:
            continue
        metrics = _review_metrics(group_rows)
        audit_rows.append(
            {
                "field": field,
                "value": value,
                **metrics,
                "top_finding_labels": _label_counts(group_rows),
            }
        )
    return sorted(
        audit_rows,
        key=lambda item: (
            -float(item.get("disagreement_rate", 0.0)),
            -float(item.get("low_confidence_rate", 0.0)),
            -int(item.get("n", 0)),
            str(item.get("field")),
            str(item.get("value")),
        ),
    )


def _failure_score(row: dict[str, Any]) -> tuple[int, float, int]:
    score = 0
    if row.get("latest_decision") == "disagree":
        score += 8
    if row.get("low_confidence") is True:
        score += 5
    if row.get("status") == "flagged":
        score += 4
    if row.get("status") == "deferred":
        score += 2
    if row.get("priority") == "urgent":
        score += 2
    probability = row.get("max_probability")
    try:
        confidence_gap = abs(float(probability) - 0.5) if probability is not None else 0.0
    except (TypeError, ValueError):
        confidence_gap = 0.0
    return (score, -confidence_gap, int(row.get("case_id") or 0))


def select_failure_cases(rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    """Return high-value cases for qualitative error review."""

    candidates = [
        row
        for row in rows
        if row.get("latest_decision") in {"disagree", "uncertain", "defer", "flag"}
        or row.get("low_confidence") is True
        or row.get("status") in {"flagged", "deferred"}
    ]
    selected = sorted(candidates, key=_failure_score, reverse=True)[:limit]
    compact: list[dict[str, Any]] = []
    for row in selected:
        compact.append(
            {
                "case_id": row.get("case_id"),
                "image_id": row.get("image_id"),
                "source_filename": row.get("source_filename"),
                "status": row.get("status"),
                "priority": row.get("priority"),
                "model_name": row.get("model_name"),
                "top_finding_labels": row.get("top_finding_labels") or [],
                "max_probability": row.get("max_probability"),
                "low_confidence": row.get("low_confidence"),
                "review_count": row.get("review_count"),
                "latest_decision": row.get("latest_decision") or "unreviewed",
                "latest_review_notes": row.get("latest_review_notes"),
                "tags": row.get("tags") or [],
            }
        )
    return compact


def build_monitoring_snapshot(
    *,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 10_000,
    drift_baseline: dict[str, Any] | None = None,
    drift_threshold: float = 0.15,
    subgroup_fields: list[str] | None = None,
    subgroup_min_cases: int = 1,
    failure_case_limit: int = 20,
) -> dict[str, Any]:
    """Build a compact snapshot of case quality and model workflow health."""

    rows = build_case_export_rows(limit=limit, store=store, db_path=db_path)
    metrics = _review_metrics(rows)
    total = metrics["n"]
    reviewed_count = metrics["reviewed"]

    labels = _label_counts(rows)
    snapshot: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "total_cases": total,
        "reviewed_cases": reviewed_count,
        "rates": {
            "review_rate": metrics["review_rate"],
            "low_confidence_rate": metrics["low_confidence_rate"],
            "disagreement_rate": metrics["disagreement_rate"],
            "flagged_rate": metrics["flagged_rate"],
            "deferred_rate": metrics["deferred_rate"],
            "urgent_rate": metrics["urgent_rate"],
        },
        "counts": {
            "low_confidence": metrics["low_confidence"],
            "disagreements": metrics["disagreements"],
            "flagged": metrics["flagged"],
            "deferred": metrics["deferred"],
            "urgent": metrics["urgent"],
            "status": _status_counts(rows),
            "latest_decision": _decision_counts(rows),
            "top_finding_labels": labels,
        },
        "subgroups": build_subgroup_audit(rows, fields=subgroup_fields, min_cases=subgroup_min_cases),
        "failure_cases": select_failure_cases(rows, limit=failure_case_limit),
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

    for subgroup in snapshot["subgroups"]:
        if subgroup["n"] >= max(subgroup_min_cases, 3) and subgroup["disagreement_rate"] >= 0.30:
            alerts.append(
                {
                    "level": "warning",
                    "type": "subgroup_disagreement",
                    "field": subgroup["field"],
                    "value": subgroup["value"],
                    "message": f"Subgroup {subgroup['field']}={subgroup['value']} has high reviewer disagreement.",
                }
            )
        if subgroup["n"] >= max(subgroup_min_cases, 3) and subgroup["low_confidence_rate"] >= 0.40:
            alerts.append(
                {
                    "level": "warning",
                    "type": "subgroup_low_confidence",
                    "field": subgroup["field"],
                    "value": subgroup["value"],
                    "message": f"Subgroup {subgroup['field']}={subgroup['value']} has high low-confidence rate.",
                }
            )

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


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body]) + "\n"


def build_monitoring_markdown(snapshot: dict[str, Any]) -> str:
    """Render a monitoring snapshot as a human-readable markdown report."""

    rates = snapshot.get("rates", {}) or {}
    counts = snapshot.get("counts", {}) or {}
    lines = [
        "# XRayMind Validation Monitoring Report",
        "",
        f"Generated: {snapshot.get('created_at', 'unknown')}",
        "",
        "> Research workflow monitoring only. Not for clinical triage.",
        "",
        "## Summary",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Total cases", snapshot.get("total_cases", 0)],
                ["Reviewed cases", snapshot.get("reviewed_cases", 0)],
                ["Review rate", f"{rates.get('review_rate', 0.0):.3f}"],
                ["Low-confidence rate", f"{rates.get('low_confidence_rate', 0.0):.3f}"],
                ["Reviewer disagreement rate", f"{rates.get('disagreement_rate', 0.0):.3f}"],
                ["Flagged rate", f"{rates.get('flagged_rate', 0.0):.3f}"],
                ["Deferred rate", f"{rates.get('deferred_rate', 0.0):.3f}"],
            ],
        ),
        "## Alerts",
        "",
    ]
    alerts = snapshot.get("alerts", []) or []
    if alerts:
        for alert in alerts:
            lines.append(f"- **{alert.get('type', 'alert')}**: {alert.get('message', '')}")
    else:
        lines.append("_No alerts triggered._")

    subgroups = snapshot.get("subgroups", []) or []
    top_subgroups = subgroups[:10]
    lines.extend(
        [
            "",
            "## Highest-risk subgroups",
            "",
            _markdown_table(
                ["Field", "Value", "n", "Review", "Low conf.", "Disagree"],
                [
                    [
                        row.get("field"),
                        row.get("value"),
                        row.get("n"),
                        f"{row.get('review_rate', 0.0):.3f}",
                        f"{row.get('low_confidence_rate', 0.0):.3f}",
                        f"{row.get('disagreement_rate', 0.0):.3f}",
                    ]
                    for row in top_subgroups
                ],
            ),
            "## Failure-case gallery",
            "",
            _markdown_table(
                ["Case", "Priority", "Status", "Decision", "Low conf.", "Top labels", "Notes"],
                [
                    [
                        row.get("case_id"),
                        row.get("priority"),
                        row.get("status"),
                        row.get("latest_decision"),
                        row.get("low_confidence"),
                        ", ".join(row.get("top_finding_labels") or []),
                        (row.get("latest_review_notes") or "")[:120],
                    ]
                    for row in (snapshot.get("failure_cases", []) or [])[:20]
                ],
            ),
            "## Label distribution",
            "",
            _markdown_table(
                ["Label", "Count"],
                [[label, count] for label, count in list((counts.get("top_finding_labels") or {}).items())[:20]],
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def save_monitoring_snapshot(
    out: str | Path = "outputs/monitoring/snapshot.json",
    *,
    baseline: str | Path | None = None,
    store: SQLiteStore | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 10_000,
    drift_threshold: float = 0.15,
    markdown_out: str | Path | None = None,
    subgroup_min_cases: int = 1,
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
        subgroup_min_cases=subgroup_min_cases,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if markdown_out:
        report_path = Path(markdown_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_monitoring_markdown(snapshot), encoding="utf-8")
        snapshot["files"] = {"json": str(out_path), "markdown": str(report_path)}
        out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return snapshot
