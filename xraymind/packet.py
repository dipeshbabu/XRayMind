"""Study packet generation for XRayMind."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from .explainability import compute_attribution, save_heatmap
from .inference import predict_image, save_prediction
from .pdf import maybe_html_to_pdf
from .report import save_html_report
from .visualization import save_heatmap_overlay, save_original_preview, save_side_by_side


def create_study_packet(
    image,
    output_dir: str | Path,
    model_name: str,
    label: Optional[str] = None,
    top_k: int = 5,
    threshold: float = 0.5,
    method: str = "integrated_gradients",
    make_pdf: bool = False,
    image_id: Optional[str] = None,
) -> dict:
    """Create a complete research packet for a single X-ray image."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction = predict_image(
        image=image, model_name=model_name, top_k=top_k, threshold=threshold
    )
    target = label or (
        prediction["top_findings"][0]["label"] if prediction.get("top_findings") else None
    )

    json_path = save_prediction(prediction, out_dir / "prediction.json")
    original_path = save_original_preview(image, out_dir / "original_preview.png")

    heatmap_path = None
    overlay_path = None
    side_by_side_path = None
    if target:
        heatmap = compute_attribution(
            image=image, label=target, model_name=model_name, method=method
        )
        heatmap_path = save_heatmap(heatmap, out_dir / "heatmap.png")
        overlay_path = save_heatmap_overlay(image, heatmap, out_dir / "overlay.png")
        side_by_side_path = save_side_by_side(
            original_path, overlay_path, out_dir / "side_by_side.png"
        )

    html_path = save_html_report(
        prediction,
        out_dir / "report.html",
        heatmap_path=heatmap_path,
        original_path=original_path,
        overlay_path=overlay_path,
        side_by_side_path=side_by_side_path,
        image_id=image_id,
    )

    pdf_path = maybe_html_to_pdf(html_path, out_dir / "report.pdf" if make_pdf else None)

    manifest = {
        "image_id": image_id,
        "model": model_name,
        "target_label": target,
        "method": method,
        "files": {
            "prediction_json": str(json_path),
            "original_preview": str(original_path),
            "heatmap": str(heatmap_path) if heatmap_path else None,
            "overlay": str(overlay_path) if overlay_path else None,
            "side_by_side": str(side_by_side_path) if side_by_side_path else None,
            "html_report": str(html_path),
            "pdf_report": str(pdf_path) if pdf_path else None,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["files"]["manifest"] = str(manifest_path)

    archive_path = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    manifest["files"]["zip"] = archive_path
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
