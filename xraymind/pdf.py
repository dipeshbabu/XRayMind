"""Optional PDF export utilities for XRayMind reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def html_to_pdf(html_path: str | Path, pdf_path: str | Path) -> Path:
    """Convert an HTML report to PDF using WeasyPrint when installed.

    WeasyPrint is optional because it may require system libraries depending on
    the environment. If unavailable, the caller receives a clear installation
    error instead of silently failing.
    """

    try:
        from weasyprint import HTML
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "PDF export requires the optional dependency 'weasyprint'. Install it with: pip install 'xraymind[pdf]' or pip install weasyprint"
        ) from exc

    html_path = Path(html_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    return pdf_path


def maybe_html_to_pdf(
    html_path: str | Path,
    pdf_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """Convert HTML to PDF only when an output path is provided."""

    if pdf_path is None:
        return None
    return html_to_pdf(html_path, pdf_path)
