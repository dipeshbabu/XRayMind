# XRayMind v0.4 Reporting Workflow

v0.4 turns the single-image demo into a cleaner research reporting workflow. It generates original-image previews, explanation heatmaps, heatmap overlays, side-by-side panels, richer HTML reports, optional PDF reports, manifests, and ZIP study packets.

> XRayMind is still a research prototype only. The generated report is not a clinical report and must not be used for diagnosis, treatment, or triage.

## 1. Create a complete packet from the CLI

```bash
xraymind packet \
  --image path/to/chest_xray.png \
  --out-dir outputs/study_packet \
  --top-k 5
```

This creates:

```text
outputs/study_packet/prediction.json
outputs/study_packet/original_preview.png
outputs/study_packet/heatmap.png
outputs/study_packet/overlay.png
outputs/study_packet/side_by_side.png
outputs/study_packet/report.html
outputs/study_packet/manifest.json
outputs/study_packet.zip
```

## 2. Optional PDF export

PDF export uses WeasyPrint, which may require system packages depending on your OS.

Install optional dependency:

```bash
pip install "xraymind[pdf]"
```

or:

```bash
pip install weasyprint
```

Then run:

```bash
xraymind packet \
  --image path/to/chest_xray.png \
  --out-dir outputs/study_packet \
  --pdf
```

This adds:

```text
outputs/study_packet/report.pdf
```

## 3. Create a packet with a specific target label

```bash
xraymind packet \
  --image path/to/chest_xray.png \
  --label Cardiomegaly \
  --out-dir outputs/cardiomegaly_packet
```

If no label is provided, XRayMind explains the highest-scoring label.

## 4. Use the script version

```bash
python scripts/create_study_packet.py \
  --image path/to/chest_xray.png \
  --out-dir outputs/study_packet \
  --top-k 5
```

## 5. Launch the demo

```bash
python app.py
```

The Gradio app now returns:

- Prediction table
- Original + explanation side-by-side image
- Explanation overlay
- Downloadable HTML report
- Downloadable ZIP packet

## 6. Report sections

The HTML report includes:

- Safety and responsible-use notice
- Study metadata
- Model name and decision threshold
- Highest-scoring finding
- Confidence band
- Low-confidence flag
- Top finding table
- Original image / explanation visualizations
- Interpretation caveats
- Recommended research checks before serious deployment

## 7. Why this matters

The point of v0.4 is to make XRayMind feel less like a raw demo and more like a reproducible research artifact. A useful research report should not only output model probabilities; it should also show what was analyzed, what model was used, what explanation was generated, and what caveats apply.

## 8. Next recommended version

v0.5 should focus on deployment/data hardening:

- DICOM ingestion
- DICOM metadata redaction
- FastAPI batch inference endpoint
- Dockerfile
- Audit logging
- Basic authentication for hosted demos
