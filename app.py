"""Modern Gradio demo for XRayMind."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from xraymind.config import DEFAULT_MODEL_NAME, DISCLAIMER, MODEL_CHOICES
from xraymind.packet import create_study_packet


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

    prediction = __import__("json").loads(Path(manifest["files"]["prediction_json"]).read_text())
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


with gr.Blocks(title="XRayMind") as demo:
    gr.Markdown(
        f"""
        # XRayMind
        Explainable chest X-ray inference, reliability-aware reporting, and downloadable study packets for research demos.

        **Safety note:** {DISCLAIMER}
        """
    )
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


if __name__ == "__main__":
    demo.launch()
