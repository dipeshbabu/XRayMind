"""HTML report generation for XRayMind predictions."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DISCLAIMER


def _confidence_band(probability: float) -> str:
    if probability >= 0.75:
        return "high model score"
    if probability >= 0.5:
        return "moderate model score"
    if probability >= 0.25:
        return "low-to-moderate model score"
    return "low model score"


def _finding_rows(prediction: Dict[str, Any]) -> str:
    rows = []
    for item in prediction.get("top_findings", []):
        label = html.escape(str(item.get("label", "")))
        probability = float(item.get("probability", 0.0))
        positive = "yes" if item.get("positive") else "no"
        band = _confidence_band(probability)
        rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{probability:.3f}</td>"
            f"<td>{html.escape(band)}</td>"
            f"<td>{positive}</td>"
            "</tr>"
        )
    return "".join(rows)


def prediction_to_html(
    prediction: Dict[str, Any],
    heatmap_path: Optional[str | Path] = None,
    original_path: Optional[str | Path] = None,
    overlay_path: Optional[str | Path] = None,
    side_by_side_path: Optional[str | Path] = None,
    image_id: Optional[str] = None,
    report_title: str = "XRayMind Chest X-ray Research Report",
) -> str:
    """Render a richer report from a prediction dictionary."""

    media_sections = []
    if side_by_side_path:
        media_sections.append(
            "<h2>Image and explanation overview</h2>"
            f"<img src='{html.escape(str(side_by_side_path))}' alt='Original image and explanation overlay' />"
        )
    else:
        if original_path:
            media_sections.append(
                "<h2>Original image preview</h2>"
                f"<img src='{html.escape(str(original_path))}' alt='Original chest X-ray preview' />"
            )
        if overlay_path:
            media_sections.append(
                "<h2>Explanation overlay</h2>"
                f"<img src='{html.escape(str(overlay_path))}' alt='XRayMind explanation overlay' />"
            )
        elif heatmap_path:
            media_sections.append(
                "<h2>Explanation heatmap</h2>"
                f"<img src='{html.escape(str(heatmap_path))}' alt='XRayMind heatmap' />"
            )

    top_label = "none"
    top_score = 0.0
    if prediction.get("top_findings"):
        top_label = str(prediction["top_findings"][0].get("label", "none"))
        top_score = float(prediction["top_findings"][0].get("probability", 0.0))

    uncertainty = prediction.get("uncertainty", {})
    low_confidence = uncertainty.get("low_confidence", False)
    image_line = f"<p><strong>Image ID:</strong> {html.escape(image_id)}</p>" if image_id else ""

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{html.escape(report_title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.55; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #202124; }}
    h1, h2, h3 {{ color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.65rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .notice {{ background: #fff8df; border: 1px solid #e5cf7a; padding: 1rem; border-radius: 8px; }}
    .summary {{ background: #eef6ff; border: 1px solid #b7d7ff; padding: 1rem; border-radius: 8px; }}
    .caveat {{ background: #f8f8f8; border-left: 4px solid #555; padding: 0.8rem 1rem; }}
    img {{ max-width: 760px; width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.3rem; }}
    .small {{ font-size: 0.92rem; color: #555; }}
  </style>
</head>
<body>
  <h1>{html.escape(report_title)}</h1>
  <p class=\"notice\"><strong>Important:</strong> {html.escape(DISCLAIMER)}</p>

  <h2>Study metadata</h2>
  {image_line}
  <p><strong>Model:</strong> <code>{html.escape(str(prediction.get('model', 'unknown')))}</code></p>
  <p><strong>Created:</strong> {html.escape(str(prediction.get('created_at', 'unknown')))}</p>
  <p><strong>Decision threshold:</strong> {float(prediction.get('threshold', 0.5)):.3f}</p>

  <h2>Model summary</h2>
  <div class=\"summary\">
    <p><strong>Highest-scoring label:</strong> {html.escape(top_label)} ({top_score:.3f})</p>
    <p><strong>Confidence band:</strong> {html.escape(_confidence_band(top_score))}</p>
    <p><strong>Low-confidence flag:</strong> {html.escape(str(low_confidence))}</p>
  </div>

  <h2>Top findings</h2>
  <table>
    <thead><tr><th>Label</th><th>Probability</th><th>Score band</th><th>Above threshold</th></tr></thead>
    <tbody>{_finding_rows(prediction)}</tbody>
  </table>

  {''.join(media_sections)}

  <h2>Interpretation caveats</h2>
  <div class=\"caveat\">
    <p>These outputs are model scores, not diagnoses. A high model score means the model associated the image with a label under its training distribution; it does not confirm the presence, location, severity, or clinical relevance of that finding.</p>
    <p>Explanation overlays visualize model sensitivity and may highlight confounders, artifacts, or dataset-specific features. They should not be treated as radiologist-validated localization.</p>
  </div>

  <h2>Recommended research checks before any serious deployment</h2>
  <ul>
    <li>Evaluate AUROC, AUPRC, Brier score, ECE, sensitivity, specificity, and confidence intervals on a held-out dataset.</li>
    <li>Calibrate thresholds per pathology and validate them externally.</li>
    <li>Check subgroup, scanner, hospital, and acquisition-protocol shift when metadata is available.</li>
    <li>Review false positives and false negatives with qualified clinical experts.</li>
  </ul>

  <p class=\"small\">Generated by XRayMind. Research prototype only.</p>
</body>
</html>"""


def save_html_report(
    prediction: Dict[str, Any],
    output_path: str | Path,
    heatmap_path: Optional[str | Path] = None,
    original_path: Optional[str | Path] = None,
    overlay_path: Optional[str | Path] = None,
    side_by_side_path: Optional[str | Path] = None,
    image_id: Optional[str] = None,
    report_title: str = "XRayMind Chest X-ray Research Report",
) -> Path:
    """Save a prediction report as HTML."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        prediction_to_html(
            prediction,
            heatmap_path=heatmap_path,
            original_path=original_path,
            overlay_path=overlay_path,
            side_by_side_path=side_by_side_path,
            image_id=image_id,
            report_title=report_title,
        ),
        encoding="utf-8",
    )
    return path
