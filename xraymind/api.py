"""FastAPI service for XRayMind inference and case review workflow."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from .api_auth import ApiPrincipal, auth_mode, require_admin, require_api_key, require_reviewer
from .api_schemas import (
    CaseDetailResponse,
    CaseEnvelope,
    CaseListResponse,
    PrincipalResponse,
    ReviewerQueueResponse,
    ServiceHealth,
)
from .audit import audit_prediction
from .cases import (
    add_review,
    assign_case,
    create_case_with_prediction,
    get_case_detail,
    list_cases,
    update_case_status,
)
from .config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K, DISCLAIMER
from .dashboard import cases_requiring_attention, dashboard_summary
from .export import export_cases
from .inference import predict_image
from .monitoring import build_monitoring_markdown, build_monitoring_snapshot, save_monitoring_snapshot
from .packet import create_study_packet
from .store import DEFAULT_DB_PATH, SQLiteStore

API_TITLE = "XRayMind API"
API_VERSION = "1.3.0"

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Research API for explainable, uncertainty-aware chest X-ray review workflows. Not for clinical use.",
)


def _db_path() -> str:
    return os.getenv("XRAYMIND_DB_PATH", DEFAULT_DB_PATH)


def _case_upload_dir() -> Path:
    path = Path(os.getenv("XRAYMIND_CASE_UPLOAD_DIR", "outputs/case_uploads"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_upload(upload: UploadFile, directory: Path) -> Path:
    suffix = Path(upload.filename or "image.png").suffix or ".png"
    safe_name = f"upload{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / safe_name
    with out.open("wb") as handle:
        handle.write(upload.file.read())
    return out


def _persist_upload(upload: UploadFile, case_id_hint: str | None = None) -> Path:
    suffix = Path(upload.filename or "image.png").suffix or ".png"
    stem = Path(upload.filename or "case").stem.replace(" ", "_")[:80] or "case"
    name = f"{case_id_hint + '_' if case_id_hint else ''}{stem}{suffix}"
    out = _case_upload_dir() / name
    counter = 1
    while out.exists():
        out = _case_upload_dir() / f"{stem}_{counter}{suffix}"
        counter += 1
    with out.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return out


def _parse_boolish(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@app.get("/health", response_model=ServiceHealth)
def health() -> dict:
    return {
        "status": "ok",
        "service": API_TITLE,
        "version": API_VERSION,
        "db_path": _db_path(),
        "auth_mode": auth_mode(),
        "disclaimer": DISCLAIMER,
    }


@app.get("/me", response_model=PrincipalResponse)
def current_principal(principal: ApiPrincipal = Depends(require_api_key)) -> ApiPrincipal:
    """Return the authenticated API principal without exposing secrets."""

    return principal


@app.post("/predict")
def predict_endpoint(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL_NAME),
    top_k: int = Form(DEFAULT_TOP_K),
    threshold: float = Form(0.5),
    audit_log: str = Form("outputs/audit/api.jsonl"),
    _: ApiPrincipal = Depends(require_api_key),
) -> JSONResponse:
    """Run prediction on one uploaded image or DICOM file."""

    with tempfile.TemporaryDirectory(prefix="xraymind_api_") as tmp:
        image_path = _write_upload(file, Path(tmp))
        result = predict_image(image_path, model_name=model, top_k=top_k, threshold=threshold)
        audit_prediction(image_path, model, audit_log, extra={"endpoint": "/predict"})
        return JSONResponse(result)


@app.post("/batch-predict")
def batch_predict_endpoint(
    files: List[UploadFile] = File(...),
    model: str = Form(DEFAULT_MODEL_NAME),
    top_k: int = Form(DEFAULT_TOP_K),
    threshold: float = Form(0.5),
    audit_log: str = Form("outputs/audit/api.jsonl"),
    _: ApiPrincipal = Depends(require_api_key),
) -> JSONResponse:
    """Run prediction on multiple uploaded images or DICOM files."""

    results = []
    with tempfile.TemporaryDirectory(prefix="xraymind_batch_") as tmp:
        tmpdir = Path(tmp)
        for idx, file in enumerate(files):
            case_dir = tmpdir / f"case_{idx}"
            image_path = _write_upload(file, case_dir)
            try:
                prediction = predict_image(image_path, model_name=model, top_k=top_k, threshold=threshold)
                audit_prediction(image_path, model, audit_log, extra={"endpoint": "/batch-predict", "filename": file.filename})
                results.append({"filename": file.filename, "status": "success", "prediction": prediction})
            except Exception as exc:
                audit_prediction(image_path, model, audit_log, status="error", extra={"endpoint": "/batch-predict", "filename": file.filename, "error": str(exc)})
                results.append({"filename": file.filename, "status": "error", "error": str(exc)})
    return JSONResponse({"results": results, "disclaimer": DISCLAIMER})


@app.post("/packet")
def packet_endpoint(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL_NAME),
    top_k: int = Form(DEFAULT_TOP_K),
    threshold: float = Form(0.5),
    label: Optional[str] = Form(default=None),
    make_pdf: bool = Form(False),
    audit_log: str = Form("outputs/audit/api.jsonl"),
    _: ApiPrincipal = Depends(require_api_key),
) -> FileResponse:
    """Create and return a ZIP study packet for an uploaded image."""

    output_root = Path(os.getenv("XRAYMIND_PACKET_DIR", "outputs/api_packets"))
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xraymind_packet_upload_") as tmp:
        image_path = _write_upload(file, Path(tmp))
        case_dir = output_root / Path(file.filename or "case").stem
        manifest = create_study_packet(
            image=image_path,
            output_dir=case_dir,
            model_name=model,
            label=label,
            top_k=top_k,
            threshold=threshold,
            make_pdf=make_pdf,
        )
        audit_prediction(image_path, model, audit_log, extra={"endpoint": "/packet", "filename": file.filename})
        zip_path = manifest["files"]["zip"]
    return FileResponse(zip_path, media_type="application/zip", filename=Path(zip_path).name)


@app.post("/cases")
def create_case_endpoint(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL_NAME),
    top_k: int = Form(DEFAULT_TOP_K),
    threshold: float = Form(0.5),
    image_id: Optional[str] = Form(default=None),
    priority: str = Form("routine"),
    tags: str = Form(""),
    assigned_to: Optional[str] = Form(default=None),
    due_at: Optional[str] = Form(default=None),
    needs_second_reader: str = Form("false"),
    _: ApiPrincipal = Depends(require_reviewer),
) -> JSONResponse:
    """Persist an uploaded image as a review case and run prediction."""

    image_path = _persist_upload(file)
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    try:
        result = create_case_with_prediction(
            image_path,
            image_id=image_id,
            source_filename=file.filename,
            model_name=model,
            top_k=top_k,
            threshold=threshold,
            priority=priority,
            tags=tag_list,
            assigned_to=assigned_to,
            due_at=due_at,
            needs_second_reader=_parse_boolish(needs_second_reader),
            db_path=_db_path(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({**result, "disclaimer": DISCLAIMER})


@app.get("/cases", response_model=CaseListResponse)
def list_cases_endpoint(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    assigned_to: Optional[str] = None,
    needs_second_reader: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    _: ApiPrincipal = Depends(require_api_key),
) -> dict:
    """List persisted review cases."""

    try:
        cases = list_cases(
            status=status,
            priority=priority,
            assigned_to=assigned_to,
            needs_second_reader=needs_second_reader,
            limit=limit,
            offset=offset,
            db_path=_db_path(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"cases": cases, "count": len(cases)}


@app.get("/cases/{case_id}", response_model=CaseDetailResponse)
def get_case_endpoint(case_id: int, _: ApiPrincipal = Depends(require_api_key)) -> dict:
    """Return case detail including latest prediction and review history."""

    try:
        return get_case_detail(case_id, db_path=_db_path())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/cases/{case_id}/status", response_model=CaseEnvelope)
def update_case_status_endpoint(
    case_id: int,
    status: str = Form(...),
    _: ApiPrincipal = Depends(require_reviewer),
) -> dict:
    """Update a case status."""

    try:
        case = update_case_status(case_id, status, db_path=_db_path())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"case": case}


@app.post("/cases/{case_id}/assign", response_model=CaseEnvelope)
def assign_case_endpoint(
    case_id: int,
    reviewer: Optional[str] = Form(default=None),
    due_at: Optional[str] = Form(default=None),
    needs_second_reader: Optional[bool] = Form(default=None),
    _: ApiPrincipal = Depends(require_reviewer),
) -> dict:
    """Assign or reassign a case to a reader/reviewer."""

    try:
        case = assign_case(
            case_id,
            reviewer=reviewer,
            due_at=due_at,
            needs_second_reader=needs_second_reader,
            db_path=_db_path(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"case": case}


@app.post("/cases/{case_id}/review")
def review_case_endpoint(
    case_id: int,
    decision: str = Form(...),
    reviewer: Optional[str] = Form(default=None),
    notes: Optional[str] = Form(default=None),
    final_labels_json: str = Form("{}"),
    next_status: Optional[str] = Form(default=None),
    review_round: Optional[int] = Form(default=None),
    _: ApiPrincipal = Depends(require_reviewer),
) -> JSONResponse:
    """Add a human reviewer decision for a case."""

    try:
        import json

        final_labels: dict[str, Any] = json.loads(final_labels_json or "{}")
        review = add_review(
            case_id,
            decision=decision,
            reviewer=reviewer,
            notes=notes,
            final_labels=final_labels,
            next_status=next_status,
            review_round=review_round,
            db_path=_db_path(),
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="final_labels_json must be valid JSON") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"review": review, "case_detail": get_case_detail(case_id, db_path=_db_path())})


@app.get("/reviewer/queue", response_model=ReviewerQueueResponse)
def reviewer_queue_endpoint(
    reviewer: Optional[str] = None,
    limit: int = 25,
    include_unassigned: bool = True,
    principal: ApiPrincipal = Depends(require_reviewer),
) -> dict:
    """Return a reviewer work queue ordered by risk/attention signals."""

    requested_reviewer = reviewer or principal.key_id
    store = SQLiteStore(_db_path())
    attention_cases = cases_requiring_attention(store=store, limit=max(limit * 3, limit))
    queue = []
    for item in attention_cases:
        assigned_to = item.get("assigned_to")
        if assigned_to == requested_reviewer or (include_unassigned and not assigned_to):
            queue.append(item)
        if len(queue) >= limit:
            break
    return {"cases": queue, "count": len(queue)}


@app.post("/reviewer/claim", response_model=CaseEnvelope)
def reviewer_claim_endpoint(
    case_id: int = Form(...),
    reviewer: Optional[str] = Form(default=None),
    due_at: Optional[str] = Form(default=None),
    principal: ApiPrincipal = Depends(require_reviewer),
) -> dict:
    """Claim a case for the current reviewer or the supplied reviewer name."""

    reviewer_name = reviewer or principal.key_id
    try:
        case = assign_case(case_id, reviewer=reviewer_name, due_at=due_at, db_path=_db_path())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"case": case}


@app.post("/reviewer/release", response_model=CaseEnvelope)
def reviewer_release_endpoint(
    case_id: int = Form(...),
    principal: ApiPrincipal = Depends(require_reviewer),
) -> dict:
    """Release a case assignment back to the shared queue."""

    try:
        detail = get_case_detail(case_id, db_path=_db_path())
        assigned_to = detail["case"].get("assigned_to")
        if principal.role != "admin" and assigned_to not in {None, principal.key_id}:
            raise HTTPException(status_code=403, detail="Only admins or the assigned reviewer can release this case")
        case = assign_case(case_id, reviewer=None, due_at=None, db_path=_db_path())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"case": case}


@app.get("/dashboard/summary")
def dashboard_summary_endpoint(_: ApiPrincipal = Depends(require_api_key)) -> JSONResponse:
    """Return dashboard aggregate metrics."""

    return JSONResponse(dashboard_summary(db_path=_db_path()))


@app.get("/dashboard/attention")
def dashboard_attention_endpoint(
    limit: int = 25,
    _: ApiPrincipal = Depends(require_api_key),
) -> JSONResponse:
    """Return cases that need human attention."""

    store = SQLiteStore(_db_path())
    return JSONResponse({"cases": cases_requiring_attention(store=store, limit=limit)})


@app.post("/exports/cases")
def export_cases_endpoint(
    out_dir: str = Form("outputs/exports"),
    status: Optional[str] = Form(default=None),
    priority: Optional[str] = Form(default=None),
    limit: int = Form(10_000),
    offset: int = Form(0),
    _: ApiPrincipal = Depends(require_admin),
) -> JSONResponse:
    """Export case workflow data to JSONL/CSV and return the manifest."""

    try:
        manifest = export_cases(
            out_dir,
            status=status,
            priority=priority,
            limit=limit,
            offset=offset,
            db_path=_db_path(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(manifest)


@app.get("/monitoring/snapshot")
def monitoring_snapshot_endpoint(
    limit: int = 10_000,
    subgroup_min_cases: int = 1,
    _: ApiPrincipal = Depends(require_api_key),
) -> JSONResponse:
    """Return an in-memory monitoring snapshot without writing files."""

    return JSONResponse(build_monitoring_snapshot(db_path=_db_path(), limit=limit, subgroup_min_cases=subgroup_min_cases))


@app.get("/monitoring/report.md", response_class=PlainTextResponse)
def monitoring_markdown_report_endpoint(
    limit: int = 10_000,
    subgroup_min_cases: int = 1,
    _: ApiPrincipal = Depends(require_api_key),
) -> str:
    """Return a human-readable markdown validation report."""

    snapshot = build_monitoring_snapshot(db_path=_db_path(), limit=limit, subgroup_min_cases=subgroup_min_cases)
    return build_monitoring_markdown(snapshot)


@app.post("/monitoring/snapshot")
def save_monitoring_snapshot_endpoint(
    out: str = Form("outputs/monitoring/snapshot.json"),
    markdown_out: Optional[str] = Form(default=None),
    baseline: Optional[str] = Form(default=None),
    limit: int = Form(10_000),
    drift_threshold: float = Form(0.15),
    subgroup_min_cases: int = Form(1),
    _: ApiPrincipal = Depends(require_admin),
) -> JSONResponse:
    """Persist a monitoring snapshot and optionally compare it with a baseline snapshot."""

    snapshot = save_monitoring_snapshot(
        out,
        baseline=baseline,
        db_path=_db_path(),
        limit=limit,
        drift_threshold=drift_threshold,
        markdown_out=markdown_out,
        subgroup_min_cases=subgroup_min_cases,
    )
    return JSONResponse(snapshot)
