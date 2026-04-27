"""Simple JSONL audit logging for XRayMind hosted/batch workflows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def file_sha256(path: str | Path) -> str:
    """Return SHA256 hash for a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_audit_event(
    event_type: str,
    output_path: str | Path = "outputs/audit/audit.jsonl",
    **payload: Any,
) -> Path:
    """Append an audit event as one JSON object per line."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return path


def audit_prediction(
    image_path: Optional[str | Path],
    model_name: str,
    output_path: str | Path = "outputs/audit/audit.jsonl",
    status: str = "success",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Record a prediction event without storing raw image data."""

    payload: Dict[str, Any] = {"model": model_name, "status": status}
    if image_path is not None:
        path = Path(image_path)
        payload.update(
            {
                "image_name": path.name,
                "image_sha256": file_sha256(path) if path.exists() and path.is_file() else None,
            }
        )
    if extra:
        payload.update(extra)
    return write_audit_event("prediction", output_path=output_path, **payload)
