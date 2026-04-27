# XRayMind

XRayMind is an explainable chest X-ray research prototype built around TorchXRayVision models, Captum attribution methods, and a lightweight Gradio interface. The project is intended for AI/ML research demos, model inspection, benchmarking, and educational exploration of chest radiography classifiers.

> **Important:** XRayMind is not a medical device and is not for clinical diagnosis, treatment, or triage. Outputs must be reviewed by qualified clinical professionals.

---

## What is new in v0.2

- Clean `xraymind/` Python package for reusable inference, preprocessing, explainability, and report generation.
- Command-line interface with `predict`, `explain`, `report`, and `demo` commands.
- Modern Gradio app in `app.py` for prediction, integrated-gradients heatmap generation, and downloadable HTML reports.
- Structured JSON prediction output with top findings, thresholds, uncertainty metadata, and safety disclaimer.
- HTML report generator for demos and research walkthroughs.
- Folder-level evaluation script for AUROC/AUPRC benchmarking on labeled image folders.
- Existing legacy Gradio code under `src/` is preserved for backward compatibility.

---

## Installation

```bash
git clone https://github.com/dipeshbabu/XRayMind.git
cd XRayMind
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

For GPU use, install the PyTorch build that matches your CUDA setup before installing the package.

---

## Quickstart

### Launch the demo

```bash
python app.py
```

or:

```bash
xraymind demo
```

### Run prediction on one image

```bash
xraymind predict \
  --image path/to/chest_xray.png \
  --out outputs/prediction.json \
  --top-k 5
```

### Generate an explanation heatmap

```bash
xraymind explain \
  --image path/to/chest_xray.png \
  --label Cardiomegaly \
  --out outputs/cardiomegaly_heatmap.png \
  --method integrated_gradients
```

### Create a JSON + HTML report

```bash
xraymind report \
  --image path/to/chest_xray.png \
  --json outputs/prediction.json \
  --html outputs/report.html \
  --heatmap outputs/heatmap.png
```

If `--label` is not provided, the report command explains the highest-scoring predicted label.

---

## Evaluate on a labeled folder

Expected CSV format:

```csv
image,Atelectasis,Cardiomegaly,Effusion
000001.png,0,1,0
000002.png,1,0,1
```

Run:

```bash
python scripts/evaluate_folder.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --out outputs/eval.csv
```

This produces per-label AUROC and AUPRC when both positive and negative examples are present.

---

## Project structure

```text
xraymind/
  cli.py              # command-line interface
  config.py           # shared constants and safety disclaimer
  explainability.py   # Captum attribution utilities
  inference.py        # structured prediction output
  model_loader.py     # cached TorchXRayVision model loading
  preprocessing.py    # image loading and X-ray preprocessing
  report.py           # HTML report generation
scripts/
  evaluate_folder.py  # labeled-folder benchmarking
  predict_image.py    # simple prediction script wrapper
src/
  legacy Gradio and ensemble code from the original prototype
app.py                # modern Gradio demo
```

---

## Current limitations

- Predictions rely on pretrained TorchXRayVision models and should not be interpreted as clinically validated outputs for a new deployment setting.
- Thresholds are currently simple defaults. A serious release should calibrate thresholds per dataset and pathology.
- Heatmaps show model sensitivity, not confirmed disease location.
- Evaluation currently assumes image-level binary labels, not radiologist-validated localization masks.
- No DICOM metadata cleaning, PHI removal, PACS integration, or regulatory workflow is included.

---

## Concrete roadmap

### v0.3: reliability and calibration

- Add validation-set threshold tuning per pathology.
- Add calibration metrics: ECE, Brier score, reliability diagrams.
- Add bootstrap confidence intervals for AUROC/AUPRC.
- Add model cards for each supported pretrained model.

### v0.4: clinician-facing reports

- Add report templates with evidence sections, caveats, and image provenance.
- Add side-by-side original image and heatmap overlay.
- Add export to PDF.
- Add finding-level confidence bands instead of only point scores.

### v0.5: data and deployment hardening

- Add DICOM ingestion with metadata redaction.
- Add batch inference API with FastAPI.
- Add Dockerfile and cloud deployment template.
- Add audit logging for research studies.

### v0.6: research-grade evaluation

- Benchmark on NIH ChestX-ray14, CheXpert, MIMIC-CXR, and PadChest where licenses permit.
- Compare multiple TorchXRayVision backbones and ensemble strategies.
- Add subgroup evaluation for scanner/site/domain shift when metadata is available.
- Add uncertainty methods such as test-time augmentation and ensemble variance.

---

## Responsible use

XRayMind should be framed as a research and education tool. For any real-world medical use, the project would need dataset governance, clinical validation, security review, deployment monitoring, regulatory review, and human oversight.
