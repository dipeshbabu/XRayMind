"""Modern Gradio demo for XRayMind."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from xraymind.cases import add_review, create_case_with_prediction, get_case_detail, list_cases
from xraymind.config import DEFAULT_MODEL_NAME, DISCLAIMER, MODEL_CHOICES
from xraymind.dashboard import cases_requiring_attention, dashboard_summary
from xraymind.packet import create_study_packet
from xraymind.store import DEFAULT_DB_PATH


def _save_uploaded_array(image: Any, prefix: str = "xraymind_case") -> Path:
    """Persist a Gradio image value to a temporary PNG path."""

    if image is None:
        raise ValueError("Upload a chest X-ray image first.")
    tmpdir = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    out = tmpdir / "image.png"
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out)
        return out
    if isinstance(image, Image.Image):
        image.save(out)
        return out
    path = Path(image)
    if path.exists():
        return path
    raise ValueError("Unsupported image input.")


def _findings_table(prediction: dict[str, Any] | None) -> pd.DataFrame:
    if not prediction:
        return pd.DataFrame()
    payload = prediction.get("prediction_json", prediction)
    return pd.DataFrame(payload.get("top_findings", []))


def run_xraymind(image, model_name: str, top_k: int, make_pdf: bool):
    if image is None:
        return pd.DataFrame(), None, None, None, None, "Upload a chest X-ray image first."

    tmpdir = Path(tempfile.mkdtemp(prefix="xraymind_"))
    manifest = create_study_packet(
        image=image,
        output_dir=tmpdir,
        model_name=model_name,
        top_k=int(top_k),
        make_pdf=bool(make_pdf),
    )

    prediction = json.loads(Path(manifest["files"]["prediction_json"]).read_text())
    table = pd.DataFrame(prediction["top_findings"])
    target = manifest.get("target_label")
    summary = (
        f"Top target explained: {target}. "
        f"Max probability: {prediction['uncertainty']['max_probability']:.3f}. "
        f"Low confidence: {prediction['uncertainty']['low_confidence']}."
    )
    return (
        table,
        manifest["files"].get("side_by_side"),
        manifest["files"].get("overlay"),
        manifest["files"].get("html_report"),
        manifest["files"].get("zip"),
        summary,
    )


def create_review_case(image, model_name: str, top_k: int, priority: str, tags: str):
    """Create a persisted review case from the Gradio case workflow tab."""

    if image is None:
        return "Upload a chest X-ray image first.", pd.DataFrame(), _dashboard_markdown(), pd.DataFrame()
    image_path = _save_uploaded_array(image, prefix="xraymind_review_case")
    tag_list = [tag.strip() for tag in (tags or "").split(",") if tag.strip()]
    result = create_case_with_prediction(
        image_path,
        model_name=model_name,
        top_k=int(top_k),
        priority=priority,
        tags=tag_list,
        db_path=DEFAULT_DB_PATH,
    )
    case = result["case"]
    prediction = result["prediction"]
    message = f"Created case #{case['id']} with status `{case['status']}` and priority `{case['priority']}`."
    return message, _findings_table(prediction), _dashboard_markdown(), _cases_dataframe()


def review_case(case_id: int | float | str, decision: str, reviewer: str, notes: str):
    """Add a human review from the Gradio UI."""

    if case_id in (None, ""):
        return "Enter a case id first.", _dashboard_markdown(), _cases_dataframe(), pd.DataFrame()
    try:
        numeric_case_id = int(case_id)
        add_review(
            numeric_case_id,
            decision=decision,
            reviewer=reviewer or None,
            notes=notes or None,
            db_path=DEFAULT_DB_PATH,
        )
        detail = get_case_detail(numeric_case_id, db_path=DEFAULT_DB_PATH)
        return (
            f"Saved `{decision}` review for case #{numeric_case_id}. Current status: `{detail['case']['status']}`.",
            _dashboard_markdown(),
            _cases_dataframe(),
            _findings_table(detail.get("latest_prediction")),
        )
    except Exception as exc:  # UI boundary: show readable error instead of crashing Gradio.
        return f"Could not save review: {exc}", _dashboard_markdown(), _cases_dataframe(), pd.DataFrame()


def load_case(case_id: int | float | str):
    """Load one case into the UI."""

    if case_id in (None, ""):
        return "Enter a case id first.", pd.DataFrame()
    try:
        detail = get_case_detail(int(case_id), db_path=DEFAULT_DB_PATH)
        case = detail["case"]
        latest = detail.get("latest_prediction")
        reviews = detail.get("reviews", [])
        summary = (
            f"Case #{case['id']} | status: `{case['status']}` | priority: `{case['priority']}` | "
            f"reviews: {len(reviews)} | image: `{case['image_path']}`"
        )
        return summary, _findings_table(latest)
    except Exception as exc:
        return f"Could not load case: {exc}", pd.DataFrame()


def _dashboard_markdown() -> str:
    summary = dashboard_summary(db_path=DEFAULT_DB_PATH)
    return (
        f"### Case dashboard\n"
        f"Total cases: **{summary['total_cases']}**  \n"
        f"Pending: **{summary['pending_cases']}** | Reviewed: **{summary['reviewed_cases']}** | "
        f"Deferred: **{summary['deferred_cases']}** | Flagged: **{summary['flagged_cases']}**  \n"
        f"Low-confidence cases: **{summary['low_confidence_cases']}**  \n"
        f"Reviewer disagreement rate: **{summary['reviewer_disagreement_rate']:.3f}**"
    )


def _cases_dataframe(attention_only: bool = False) -> pd.DataFrame:
    if attention_only:
        rows = cases_requiring_attention(db_path=DEFAULT_DB_PATH, limit=50)
    else:
        rows = list_cases(limit=50, db_path=DEFAULT_DB_PATH)
    if not rows:
        return pd.DataFrame(columns=["id", "status", "priority", "model_name", "image_path", "created_at"])
    keep = ["id", "status", "priority", "model_name", "image_path", "created_at", "updated_at"]
    return pd.DataFrame([{key: row.get(key) for key in keep} for row in rows])


def refresh_dashboard(attention_only: bool):
    return _dashboard_markdown(), _cases_dataframe(attention_only=bool(attention_only))


with gr.Blocks(title="XRayMind") as demo:
    gr.Markdown(
        f"""
        # XRayMind
        Explainable chest X-ray inference, reliability-aware reporting, downloadable study packets, and local case review workflows for research demos.

        **Safety note:** {DISCLAIMER}
        """
    )

    with gr.Tab("Single-image packet"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Chest X-ray", type="numpy")
                model = gr.Dropdown(
                    label="Model", choices=list(MODEL_CHOICES), value=DEFAULT_MODEL_NAME
                )
                top_k = gr.Slider(label="Top K findings", minimum=1, maximum=10, value=5, step=1)
                make_pdf = gr.Checkbox(label="Try PDF export if WeasyPrint is installed", value=False)
                button = gr.Button("Run XRayMind")
            with gr.Column():
                summary = gr.Textbox(label="Summary")
                predictions = gr.Dataframe(label="Top findings")
                side_by_side = gr.Image(label="Original + explanation overview")
                overlay = gr.Image(label="Explanation overlay")
                report = gr.File(label="HTML report")
                packet = gr.File(label="Study packet ZIP")

        button.click(
            run_xraymind,
            inputs=[image, model, top_k, make_pdf],
            outputs=[predictions, side_by_side, overlay, report, packet, summary],
        )

    with gr.Tab("Case review workflow"):
        gr.Markdown("Create local review cases, capture reviewer decisions, and monitor the queue from a SQLite-backed workflow store.")
        with gr.Row():
            with gr.Column():
                case_image = gr.Image(label="Chest X-ray for review case", type="numpy")
                case_model = gr.Dropdown(label="Model", choices=list(MODEL_CHOICES), value=DEFAULT_MODEL_NAME)
                case_top_k = gr.Slider(label="Top K findings", minimum=1, maximum=10, value=5, step=1)
                case_priority = gr.Dropdown(label="Priority", choices=["routine", "elevated", "urgent"], value="routine")
                case_tags = gr.Textbox(label="Tags", placeholder="demo, review")
                create_case_button = gr.Button("Create case + prediction")
                case_message = gr.Markdown()
            with gr.Column():
                case_findings = gr.Dataframe(label="Latest prediction")
                dashboard_md = gr.Markdown(value=_dashboard_markdown())
                attention_only = gr.Checkbox(label="Show attention queue only", value=False)
                refresh_button = gr.Button("Refresh dashboard")

        with gr.Row():
            case_table = gr.Dataframe(label="Cases", value=_cases_dataframe())

        with gr.Row():
            with gr.Column():
                selected_case_id = gr.Number(label="Case ID", precision=0)
                load_case_button = gr.Button("Load case")
                loaded_case_summary = gr.Markdown()
                loaded_case_findings = gr.Dataframe(label="Loaded case prediction")
            with gr.Column():
                review_decision = gr.Dropdown(
                    label="Review decision",
                    choices=["agree", "disagree", "uncertain", "defer", "flag"],
                    value="uncertain",
                )
                reviewer = gr.Textbox(label="Reviewer", placeholder="reviewer_a")
                notes = gr.Textbox(label="Notes", lines=4)
                review_button = gr.Button("Save review")
                review_message = gr.Markdown()

        create_case_button.click(
            create_review_case,
            inputs=[case_image, case_model, case_top_k, case_priority, case_tags],
            outputs=[case_message, case_findings, dashboard_md, case_table],
        )
        refresh_button.click(
            refresh_dashboard,
            inputs=[attention_only],
            outputs=[dashboard_md, case_table],
        )
        load_case_button.click(
            load_case,
            inputs=[selected_case_id],
            outputs=[loaded_case_summary, loaded_case_findings],
        )
        review_button.click(
            review_case,
            inputs=[selected_case_id, review_decision, reviewer, notes],
            outputs=[review_message, dashboard_md, case_table, loaded_case_findings],
        )


if __name__ == "__main__":
    demo.launch()
