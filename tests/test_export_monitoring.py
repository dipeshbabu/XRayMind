from __future__ import annotations

import json
from pathlib import Path

from xraymind.cases import add_review, create_case, save_prediction_for_case
from xraymind.export import build_case_export_rows, export_cases
from xraymind.monitoring import build_monitoring_snapshot, save_monitoring_snapshot
from xraymind.store import SQLiteStore


def _prediction(label: str = "Atelectasis", prob: float = 0.91, low_confidence: bool = False) -> dict:
    return {
        "top_findings": [{"label": label, "probability": prob}],
        "all_probabilities": {label: prob},
        "uncertainty": {"max_probability": prob, "low_confidence": low_confidence},
        "disclaimer": "test",
    }


def test_export_cases_writes_jsonl_csv_and_manifest(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cases.sqlite3")
    case = create_case("sample.png", image_id="img-1", priority="urgent", tags=["demo"], store=store)
    save_prediction_for_case(
        case["id"],
        _prediction(low_confidence=True),
        model_name="test-model",
        threshold=0.5,
        top_k=3,
        store=store,
    )
    add_review(case["id"], decision="disagree", reviewer="r1", final_labels={"Atelectasis": False}, store=store)

    rows = build_case_export_rows(store=store)
    assert len(rows) == 1
    assert rows[0]["top_finding_labels"] == ["Atelectasis"]
    assert rows[0]["latest_decision"] == "disagree"

    manifest = export_cases(tmp_path / "exports", store=store)
    assert manifest["row_count"] == 1
    assert Path(manifest["files"]["jsonl"]).exists()
    assert Path(manifest["files"]["csv"]).exists()
    assert Path(manifest["files"]["manifest"]).exists()
    assert manifest["sha256"]["jsonl"]


def test_monitoring_snapshot_counts_alerts_and_drift(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cases.sqlite3")
    for idx in range(6):
        case = create_case(f"sample-{idx}.png", priority="urgent" if idx == 0 else "routine", store=store)
        save_prediction_for_case(
            case["id"],
            _prediction(prob=0.2 if idx < 2 else 0.9, low_confidence=idx < 2),
            model_name="test-model",
            threshold=0.5,
            top_k=3,
            store=store,
        )
        add_review(case["id"], decision="disagree" if idx < 2 else "agree", reviewer="r1", store=store)

    snapshot = build_monitoring_snapshot(store=store)
    assert snapshot["total_cases"] == 6
    assert snapshot["reviewed_cases"] == 6
    assert snapshot["counts"]["disagreements"] == 2
    assert snapshot["rates"]["disagreement_rate"] > 0.2
    assert any(alert["type"] == "review_disagreement" for alert in snapshot["alerts"])

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"created_at": "baseline", "rates": {"low_confidence_rate": 0.0}}), encoding="utf-8")
    saved = save_monitoring_snapshot(tmp_path / "snapshot.json", baseline=baseline, store=store, drift_threshold=0.1)
    assert Path(tmp_path / "snapshot.json").exists()
    assert "drift" in saved
    assert any(alert["type"] == "rate_drift" for alert in saved["alerts"])
