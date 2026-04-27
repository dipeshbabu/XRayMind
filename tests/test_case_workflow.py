from __future__ import annotations

import sqlite3
from pathlib import Path

from xraymind.cases import (
    add_review,
    assign_case,
    create_case,
    get_case_detail,
    list_cases,
    save_prediction_for_case,
    update_case_status,
)
from xraymind.dashboard import cases_requiring_attention, dashboard_summary
from xraymind.export import build_case_export_rows
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
        assigned_to="reviewer_a",
        store=store,
    )
    assert case["id"] == 1
    assert case["status"] == "pending"
    assert case["priority"] == "elevated"
    assert case["tags"] == ["demo", "pytest"]
    assert case["assigned_to"] == "reviewer_a"

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
    assert review["review_round"] == 1

    detail = get_case_detail(case["id"], store=store)
    assert detail["case"]["status"] == "reviewed"
    assert detail["case"]["needs_second_reader"] is True
    assert len(detail["reviews"]) == 1
    assert detail["latest_prediction"]["prediction_json"]["uncertainty"]["low_confidence"] is True

    summary = dashboard_summary(store=store)
    assert summary["total_cases"] == 1
    assert summary["reviewed_cases"] == 1
    assert summary["low_confidence_cases"] == 1
    assert summary["second_reader_cases"] == 1
    assert summary["assignment_counts"] == {"reviewer_a": 1}
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


def test_assign_case_and_export_fields(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "workflow.sqlite3")
    case = create_case(tmp_path / "demo.png", priority="urgent", store=store)

    assigned = assign_case(
        case["id"],
        reviewer="reader_b",
        due_at="2026-05-01T17:00:00Z",
        needs_second_reader=True,
        store=store,
    )
    assert assigned["assigned_to"] == "reader_b"
    assert assigned["due_at"] == "2026-05-01T17:00:00Z"
    assert assigned["needs_second_reader"] is True

    add_review(case["id"], decision="agree", reviewer="reader_b", store=store)
    rows = build_case_export_rows(store=store)
    assert rows[0]["assigned_to"] == "reader_b"
    assert rows[0]["due_at"] == "2026-05-01T17:00:00Z"
    assert rows[0]["needs_second_reader"] is False
    assert rows[0]["latest_reviewer"] == "reader_b"


def test_store_migrates_existing_v1_database(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                image_id TEXT,
                source_filename TEXT,
                model_name TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                priority TEXT NOT NULL DEFAULT 'routine',
                patient_context TEXT,
                tags TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                threshold REAL NOT NULL,
                top_k INTEGER NOT NULL,
                prediction_json TEXT NOT NULL,
                max_probability REAL,
                low_confidence INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                reviewer TEXT,
                decision TEXT NOT NULL,
                notes TEXT,
                final_labels TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER,
                event_type TEXT NOT NULL,
                payload TEXT,
                created_at TEXT NOT NULL
            );
            """
        )

    store = SQLiteStore(db)
    case = create_case("after_migration.png", assigned_to="reader_m", needs_second_reader=True, store=store)

    assert case["assigned_to"] == "reader_m"
    assert case["needs_second_reader"] is True
    columns = {row["name"] for row in store.fetch_all("PRAGMA table_info(cases)")}
    assert {"assigned_to", "due_at", "needs_second_reader"}.issubset(columns)
    review_columns = {row["name"] for row in store.fetch_all("PRAGMA table_info(reviews)")}
    assert "review_round" in review_columns
