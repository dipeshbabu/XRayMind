"""HTML report generation for XRayMind predictions."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DISCLAIMER


def prediction_to_html(
    prediction: Dict[str, Any],
    heatmap_path: Optional[str | Path] = None,
) -> str:
    """Render a compact HTML report from a prediction dictionary."""

    rows = []
    for item in prediction.get("top_findings", []):
        label = html.escape(str(item.get("label", "")))
        probability = float(item.get("probability", 0.0))
        positive = "yes" if item.get("positive") else "no"
        rows.append(
            f"<tr><td>{label}</td><td>{probability:.3f}</td><td>{positive}</td></tr>"
        )

    heatmap_html = ""
    if heatmap_path:
        heatmap_html = (
            "<h2>Explanation heatmap</h2>"
            f"<img src='{html.escape(str(heatmap_path))}' alt='XRayMind heatmap' />"
        )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>XRayMind Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.5; max-width: 920px; margin: 2rem auto; padding: 0 1rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.6rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .notice {{ background: #fff8df; border: 1px solid #e5cf7a; padding: 1rem; border-radius: 8px; }}
    img {{ max-width: 512px; width: 100%; border: 1px solid #ddd; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.3rem; }}
  </style>
</head>
<body>
  <h1>XRayMind Chest X-ray Research Report</h1>
  <p class=\"notice\"><strong>Important:</strong> {html.escape(DISCLAIMER)}</p>
  <p><strong>Model:</strong> <code>{html.escape(str(prediction.get('model', 'unknown')))}</code></p>
  <p><strong>Created:</strong> {html.escape(str(prediction.get('created_at', 'unknown')))}</p>
  <h2>Top findings</h2>
  <table>
    <thead><tr><th>Label</th><th>Probability</th><th>Above threshold</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {heatmap_html}
</body>
</html>"""


def save_html_report(
    prediction: Dict[str, Any],
    output_path: str | Path,
    heatmap_path: Optional[str | Path] = None,
) -> Path:
    """Save a prediction report as HTML."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prediction_to_html(prediction, heatmap_path), encoding="utf-8")
    return path
