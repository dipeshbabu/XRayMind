from __future__ import annotations

from pathlib import Path

from xraymind.cases import (
    add_review,
    create_case,
    get_case_detail,
    list_cases,
    save_prediction_for_case,
    update_case_status,
)
from xraymind.dashboard import cases_requiring_attention, dashboard_summary
from xraymind.store import SQLiteStore


def _prediction_payload(max_probability: float = 0.42, low_confidence: bool = True) -> dict:
    return {
        "top_findings": [
            {"label": "Cardiomegaly", "probability": max_probability, "present": False},
            {"label": "Effusion", "probability": 0.31, "present": False},
        ],
        "uncertainty": {
            "max_probability": max_probability,
            "low_confidence": low_confidence,
        },
    }


def test_case_prediction_review_dashboard_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "workflow.sqlite3"
    store = SQLiteStore(db_path)

    case = create_case(
        tmp_path / "case.png",
        image_id="demo-001",
        model_name="demo-model",
        priority="elevated",
        tags=["demo", "pytest"],
        store=store,
    )
    assert case["id"] == 1
    assert case["status"] == "pending"
    assert case["priority"] == "elevated"
    assert case["tags"] == ["demo", "pytest"]

    prediction = save_prediction_for_case(
        case["id"],
        _prediction_payload(),
        model_name="demo-model",
        threshold=0.5,
        top_k=2,
        store=store,
    )
    assert prediction["low_confidence"] is True
    assert prediction["prediction_json"]["top_findings"][0]["label"] == "Cardiomegaly"

    review = add_review(
        case["id"],
        decision="uncertain",
        reviewer="reviewer_a",
        notes="Needs second read.",
        final_labels={"Cardiomegaly": "uncertain"},
        store=store,
    )
    assert review["decision"] == "uncertain"

    detail = get_case_detail(case["id"], store=store)
    assert detail["case"]["status"] == "reviewed"
    assert len(detail["reviews"]) == 1
    assert detail["latest_prediction"]["prediction_json"]["uncertainty"]["low_confidence"] is True

    summary = dashboard_summary(store=store)
    assert summary["total_cases"] == 1
    assert summary["reviewed_cases"] == 1
    assert summary["low_confidence_cases"] == 1
    assert summary["review_decision_counts"] == {"uncertain": 1}
    assert summary["top_alert_labels"][0]["label"] == "Cardiomegaly"


def test_list_cases_status_update_and_attention_queue(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "workflow.sqlite3")
    routine = create_case(tmp_path / "routine.png", priority="routine", store=store)
    urgent = create_case(tmp_path / "urgent.png", priority="urgent", store=store)

    save_prediction_for_case(
        routine["id"],
        _prediction_payload(max_probability=0.91, low_confidence=False),
        model_name="demo-model",
        threshold=0.5,
        top_k=2,
        store=store,
    )
    save_prediction_for_case(
        urgent["id"],
        _prediction_payload(max_probability=0.25, low_confidence=True),
        model_name="demo-model",
        threshold=0.5,
        top_k=2,
        store=store,
    )

    updated = update_case_status(urgent["id"], "flagged", store=store)
    assert updated["status"] == "flagged"

    pending_cases = list_cases(status="pending", store=store)
    assert [case["id"] for case in pending_cases] == [routine["id"]]

    attention = cases_requiring_attention(store=store)
    assert attention[0]["id"] == urgent["id"]
    assert attention[0]["priority"] == "urgent"
    assert attention[0]["low_confidence"] == 1
