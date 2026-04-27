"""FastAPI service for XRayMind batch and single-image inference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .audit import audit_prediction
from .config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K, DISCLAIMER
from .inference import predict_image
from .packet import create_study_packet

API_TITLE = "XRayMind API"
API_VERSION = "0.5.0"

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Research API for explainable chest X-ray inference. Not for clinical use.",
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Require X-API-Key only when XRAYMIND_API_KEY is configured."""

    expected = os.getenv("XRAYMIND_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _write_upload(upload: UploadFile, directory: Path) -> Path:
    suffix = Path(upload.filename or "image.png").suffix or ".png"
    safe_name = f"upload{suffix}"
    out = directory / safe_name
    with out.open("wb") as handle:
        handle.write(upload.file.read())
    return out


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": API_TITLE,
        "version": API_VERSION,
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict")
def predict_endpoint(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL_NAME),
    top_k: int = Form(DEFAULT_TOP_K),
    threshold: float = Form(0.5),
    audit_log: str = Form("outputs/audit/api.jsonl"),
    _: None = Depends(require_api_key),
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
    _: None = Depends(require_api_key),
) -> JSONResponse:
    """Run prediction on multiple uploaded images or DICOM files."""

    results = []
    with tempfile.TemporaryDirectory(prefix="xraymind_batch_") as tmp:
        tmpdir = Path(tmp)
        for idx, file in enumerate(files):
            image_path = _write_upload(file, tmpdir / f"case_{idx}") if False else None
            case_dir = tmpdir / f"case_{idx}"
            case_dir.mkdir(parents=True, exist_ok=True)
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
    _: None = Depends(require_api_key),
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
