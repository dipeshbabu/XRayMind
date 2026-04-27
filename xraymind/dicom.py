"""Optional DICOM ingestion and metadata redaction helpers for XRayMind.

This module is intentionally dependency-light at import time. pydicom is only
required when DICOM functions are called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from PIL import Image

PHI_TAGS: tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "AccessionNumber",
    "StudyID",
)

SAFE_METADATA_TAGS: tuple[str, ...] = (
    "Modality",
    "StudyDate",
    "SeriesDate",
    "ViewPosition",
    "BodyPartExamined",
    "Manufacturer",
    "ManufacturerModelName",
    "Rows",
    "Columns",
    "PhotometricInterpretation",
    "PixelSpacing",
)


def _require_pydicom():
    try:
        import pydicom
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "DICOM support requires the optional dependency 'pydicom'. Install it with: pip install 'xraymind[dicom]' or pip install pydicom"
        ) from exc
    return pydicom


def read_dicom(path: str | Path):
    """Read a DICOM file using pydicom."""

    pydicom = _require_pydicom()
    return pydicom.dcmread(str(path))


def dicom_to_array(path: str | Path) -> np.ndarray:
    """Convert a DICOM image to an 8-bit grayscale numpy array."""

    ds = read_dicom(path)
    arr = ds.pixel_array.astype(float)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    arr = arr - np.nanmin(arr)
    arr = arr / (np.nanmax(arr) + 1e-8)
    if photometric == "MONOCHROME1":
        arr = 1.0 - arr
    return (255 * arr).astype(np.uint8)


def dicom_to_png(path: str | Path, output_path: str | Path) -> Path:
    """Convert a DICOM image to a PNG preview suitable for XRayMind inference."""

    arr = dicom_to_array(path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out)
    return out


def extract_safe_metadata(path: str | Path, tags: Iterable[str] = SAFE_METADATA_TAGS) -> Dict[str, Any]:
    """Extract a small allowlist of non-direct-identifying DICOM metadata."""

    ds = read_dicom(path)
    metadata: Dict[str, Any] = {}
    for tag in tags:
        if hasattr(ds, tag):
            value = getattr(ds, tag)
            try:
                json.dumps(value)
                metadata[tag] = value
            except TypeError:
                metadata[tag] = str(value)
    return metadata


def redact_dicom(
    input_path: str | Path,
    output_path: str | Path,
    remove_private_tags: bool = True,
    replacement: str = "REDACTED",
) -> Path:
    """Write a redacted DICOM copy by replacing common PHI fields.

    This helper is a research convenience, not a complete HIPAA de-identification
    pipeline. Real deployments need a formal privacy review.
    """

    ds = read_dicom(input_path)
    if remove_private_tags:
        ds.remove_private_tags()
    for tag in PHI_TAGS:
        if hasattr(ds, tag):
            setattr(ds, tag, replacement)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out))
    return out


def write_safe_metadata_json(input_path: str | Path, output_path: str | Path) -> Path:
    """Save safe allowlisted DICOM metadata to JSON."""

    metadata = extract_safe_metadata(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def is_dicom_path(path: str | Path) -> bool:
    """Return True for common DICOM file extensions."""

    suffix = Path(path).suffix.lower()
    return suffix in {".dcm", ".dicom"}
