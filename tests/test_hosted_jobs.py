from pathlib import Path

from xraymind.hosted_jobs import (
    cancel_job,
    enqueue_case_prediction_job,
    get_job,
    list_jobs,
    process_next_job,
)
from xraymind.store import SQLiteStore


def test_enqueue_list_and_cancel_job(tmp_path: Path):
    store = SQLiteStore(tmp_path / "cases.sqlite3")
    job = enqueue_case_prediction_job(
        tmp_path / "sample.png",
        tenant_id="clinic-a",
        image_id="case-001",
        priority="urgent",
        tags=["triage"],
        store=store,
    )

    assert job["status"] == "queued"
    assert job["tenant_id"] == "clinic-a"
    assert job["payload"]["image_id"] == "case-001"
    assert job["payload"]["tags"] == ["triage"]

    jobs = list_jobs(tenant_id="clinic-a", status="queued", store=store)
    assert [item["id"] for item in jobs] == [job["id"]]

    cancelled = cancel_job(job["id"], store=store)
    assert cancelled["status"] == "cancelled"
    assert get_job(job["id"], store=store)["status"] == "cancelled"


def test_process_next_job_failure_is_recorded(tmp_path: Path):
    store = SQLiteStore(tmp_path / "cases.sqlite3")
    job = enqueue_case_prediction_job(tmp_path / "missing.png", tenant_id="clinic-a", store=store)

    processed = process_next_job(store=store)

    assert processed is not None
    assert processed["id"] == job["id"]
    assert processed["status"] == "failed"
    assert processed["attempts"] == 1
    assert processed["error"]["type"] in {"FileNotFoundError", "ValueError"}
    assert process_next_job(store=store) is None
