"""Modern Gradio demo for XRayMind v0.2."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from xraymind.config import DEFAULT_MODEL_NAME, DISCLAIMER, MODEL_CHOICES
from xraymind.explainability import explain_to_file
from xraymind.inference import predict_image
from xraymind.report import save_html_report


def run_xraymind(image, model_name: str, top_k: int):
    if image is None:
        return pd.DataFrame(), None, None, "Upload a chest X-ray image first."

    result = predict_image(image=image, model_name=model_name, top_k=int(top_k))
    table = pd.DataFrame(result["top_findings"])

    tmpdir = Path(tempfile.mkdtemp(prefix="xraymind_"))
    json_path = tmpdir / "prediction.json"
    html_path = tmpdir / "report.html"
    heatmap_path = tmpdir / "heatmap.png"

    target = result["top_findings"][0]["label"] if result["top_findings"] else None
    if target:
        explain_to_file(
            image=image,
            label=target,
            output_path=heatmap_path,
            model_name=model_name,
            method="integrated_gradients",
        )
    save_html_report(result, html_path, heatmap_path=heatmap_path if target else None)
    json_path.write_text(__import__("json").dumps(result, indent=2), encoding="utf-8")

    summary = (
        f"Top target explained: {target}. "
        f"Max probability: {result['uncertainty']['max_probability']:.3f}. "
        f"Low confidence: {result['uncertainty']['low_confidence']}."
    )
    return table, str(heatmap_path) if target else None, str(html_path), summary


with gr.Blocks(title="XRayMind") as demo:
    gr.Markdown(
        f"""
        # XRayMind
        Explainable chest X-ray inference and report generation for research demos.

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
            button = gr.Button("Run XRayMind")
        with gr.Column():
            summary = gr.Textbox(label="Summary")
            predictions = gr.Dataframe(label="Top findings")
            heatmap = gr.Image(label="Integrated gradients heatmap")
            report = gr.File(label="HTML report")

    button.click(
        run_xraymind,
        inputs=[image, model, top_k],
        outputs=[predictions, heatmap, report, summary],
    )


if __name__ == "__main__":
    demo.launch()
