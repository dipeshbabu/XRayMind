# XRayMind

XRayMind is an explainable chest X-ray research prototype built around TorchXRayVision models, Captum attribution methods, reliability evaluation, reporting, DICOM ingestion, hosted API inference, uncertainty, model cards, dataset cards, selective prediction, and a lightweight Gradio interface. The project is intended for AI/ML research demos, model inspection, benchmarking, and educational exploration of chest radiography classifiers.

> **Important:** XRayMind is not a medical device and is not for clinical diagnosis, treatment, or triage. Outputs must be reviewed by qualified clinical professionals.

---

## What is new in v0.7

- Selective prediction / abstention utilities in `xraymind/selective.py`.
- Standalone selective-risk script: `scripts/selective_prediction.py`.
- Benchmark integration with `--selective`.
- Per-label selective curves showing coverage, deferral rate, risk, accuracy, AUROC, Brier score, sensitivity, specificity, precision, and F1.
- Aggregate selective-risk summaries across labels.
- Automatic operating-point selection with `--max-risk` and `--min-coverage`.
- Markdown selective prediction reports and risk-coverage plots.
- Dedicated workflow guide in `docs/V0_7_SELECTIVE_PREDICTION.md`.

## What changed in v0.6

- Multi-model benchmark runner for folder/CSV chest X-ray datasets.
- Reusable evaluation utilities in `xraymind/evaluation.py`.
- Dataset-card generation for benchmark datasets.
- Per-model benchmark model-card generation.
- Combined leaderboard across TorchXRayVision models.
- Optional subgroup evaluation for metadata slices such as sex, site, age group, or view position.
- Test-time augmentation uncertainty helper and `xraymind tta` CLI command.
- Dedicated research workflow guide in `docs/V0_6_RESEARCH_EVAL.md`.

## What changed in v0.5

- Optional DICOM ingestion for `.dcm` / `.dicom` inputs.
- DICOM-to-PNG conversion script with safe metadata export.
- Research helper for redacting common direct-identifying DICOM fields.
- FastAPI service for single-image prediction, batch prediction, and downloadable study-packet ZIPs.
- Optional API-key protection for hosted demos via `XRAYMIND_API_KEY`.
- JSONL audit logging for CLI and API workflows without storing raw image data.
- Dockerfile and Docker Compose setup for local hosted deployment.
- Dedicated API documentation in `docs/api.md`.

## What changed in v0.4

- Study-packet generation for single-image research/demo workflows.
- Standardized original image preview.
- Heatmap overlay on the original X-ray.
- Side-by-side original image and explanation overview.
- Richer HTML report with metadata, finding-level score bands, caveats, and recommended validation checks.
- Optional PDF export through the `pdf` extra.
- Updated Gradio app with downloadable HTML report and ZIP packet.
- Dedicated reporting workflow documentation in `docs/V0_4_REPORTING.md`.

## What changed in v0.3

- Reliability metrics: Brier score, expected calibration error, sensitivity, specificity, precision, and F1.
- Threshold tuning per pathology with F1 or Youden objective.
- Bootstrap confidence intervals for AUROC and AUPRC.
- Reliability diagram generation per label.
- Model-card generator for evaluation summaries.
- Dedicated reliability workflow documentation in `docs/V0_3_RELIABILITY.md`.

---

## Installation

```bash
git clone https://github.com/dipeshbabu/XRayMind.git
cd XRayMind
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

Or install runtime dependencies directly:

```bash
pip install -r requirements.txt
```

For optional PDF export:

```bash
pip install -e ".[pdf]"
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
  --top-k 5 \
  --audit-log outputs/audit/audit.jsonl
```

DICOM files are also supported when `pydicom` is installed:

```bash
xraymind predict \
  --image path/to/case.dcm \
  --out outputs/prediction.json
```

### Run test-time augmentation uncertainty

```bash
xraymind tta \
  --image path/to/chest_xray.png \
  --model densenet121-res224-all \
  --out outputs/tta_prediction.json
```

This returns mean and standard deviation across simple augmentations. Treat it as a research uncertainty signal, not a clinical confidence score.

### Convert or redact a DICOM file

```bash
xraymind dicom \
  --dicom path/to/case.dcm \
  --png outputs/dicom/case.png \
  --metadata outputs/dicom/case_metadata.json \
  --redacted outputs/dicom/case_redacted.dcm
```

### Generate an explanation heatmap

```bash
xraymind explain \
  --image path/to/chest_xray.png \
  --label Cardiomegaly \
  --out outputs/cardiomegaly_heatmap.png \
  --method integrated_gradients
```

### Create a full study packet

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

Optional PDF:

```bash
xraymind packet \
  --image path/to/chest_xray.png \
  --out-dir outputs/study_packet \
  --pdf
```

### Run the hosted API

```bash
uvicorn xraymind.api:app --reload --host 0.0.0.0 --port 8000
```

Optional API key:

```bash
export XRAYMIND_API_KEY=change-me
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: change-me" \
  -F "file=@path/to/chest_xray.png"
```

Docker:

```bash
XRAYMIND_API_KEY=change-me docker compose up --build
```

See `docs/api.md` for the full API workflow.

### Create a JSON + HTML report

```bash
xraymind report \
  --image path/to/chest_xray.png \
  --json outputs/prediction.json \
  --html outputs/report.html \
  --heatmap outputs/heatmap.png \
  --original outputs/original_preview.png
```

If `--label` is not provided, the report command explains the highest-scoring predicted label.

See `docs/V0_4_REPORTING.md` for the full reporting workflow.

---

## Evaluate on a labeled folder

Expected CSV format:

```csv
image,Atelectasis,Cardiomegaly,Effusion,sex,view_position
000001.png,0,1,0,F,PA
000002.png,1,0,1,M,AP
```

Run a basic evaluation:

```bash
python scripts/evaluate_folder.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --out outputs/eval.csv
```

Run a multi-model research benchmark:

```bash
python scripts/benchmark_models.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all densenet121-res224-nih densenet121-res224-chex \
  --subgroups sex view_position \
  --out-dir outputs/benchmark_v0_7 \
  --save-plots \
  --selective \
  --max-risk 0.15
```

This produces:

```text
outputs/benchmark_v0_7/DATASET_CARD.md
outputs/benchmark_v0_7/leaderboard.csv
outputs/benchmark_v0_7/combined_metrics.csv
outputs/benchmark_v0_7/combined_subgroup_metrics.csv
outputs/benchmark_v0_7/combined_selective_summary.csv
outputs/benchmark_v0_7/run_manifest.json
outputs/benchmark_v0_7/<model>/predictions.csv
outputs/benchmark_v0_7/<model>/metrics.csv
outputs/benchmark_v0_7/<model>/MODEL_CARD.md
outputs/benchmark_v0_7/<model>/reliability_plots/*.png
outputs/benchmark_v0_7/<model>/selective_prediction/selective_curves.csv
outputs/benchmark_v0_7/<model>/selective_prediction/selective_summary.csv
outputs/benchmark_v0_7/<model>/selective_prediction/operating_point.json
outputs/benchmark_v0_7/<model>/selective_prediction/selective_risk_curve.png
outputs/benchmark_v0_7/<model>/selective_prediction/SELECTIVE_PREDICTION_REPORT.md
```

Run a fuller reliability evaluation for one model:

```bash
python scripts/evaluate_folder.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --out outputs/eval.csv \
  --predictions-out outputs/predictions.csv \
  --tune-thresholds \
  --threshold-objective f1 \
  --bootstrap 1000 \
  --save-plots
```

This produces per-label AUROC, AUPRC, Brier score, ECE, sensitivity, specificity, precision, F1, optional confidence intervals, and optional reliability plots.

### Run selective prediction from existing predictions

```bash
python scripts/selective_prediction.py \
  --labels data/labels.csv \
  --predictions outputs/benchmark_v0_7/densenet121-res224-all/predictions.csv \
  --model densenet121-res224-all \
  --dataset "XRayMind folder dataset" \
  --out-dir outputs/selective_v0_7 \
  --max-risk 0.15
```

This produces selective curves, an aggregate risk-coverage summary, an operating-point JSON file, a PNG plot, and a Markdown report.

### Tune thresholds after inference

```bash
python scripts/tune_thresholds.py \
  --labels data/labels.csv \
  --predictions outputs/predictions.csv \
  --out outputs/thresholds.csv \
  --objective f1
```

### Generate a model card from existing metrics

```bash
python scripts/make_model_card.py \
  --metrics outputs/eval.csv \
  --model densenet121-res224-all \
  --dataset "NIH ChestX-ray14 validation split" \
  --out outputs/MODEL_CARD.md
```

See `docs/V0_3_RELIABILITY.md`, `docs/V0_6_RESEARCH_EVAL.md`, and `docs/V0_7_SELECTIVE_PREDICTION.md` for the full reliability, benchmark, and abstention workflows.

---

## Project structure

```text
xraymind/
  api.py              # FastAPI hosted inference service
  audit.py            # JSONL audit logging helpers
  bootstrap.py        # bootstrap confidence intervals
  cli.py              # command-line interface
  config.py           # shared constants and safety disclaimer
  dicom.py            # DICOM ingestion, conversion, and redaction helpers
  evaluation.py       # reusable benchmark, dataset-card, and model-card helpers
  explainability.py   # Captum attribution utilities
  inference.py        # structured prediction output
  metrics.py          # reliability metrics and threshold tuning
  model_loader.py     # cached TorchXRayVision model loading
  packet.py           # complete study-packet generation
  pdf.py              # optional HTML-to-PDF export
  plots.py            # reliability diagrams
  preprocessing.py    # image loading and X-ray preprocessing
  report.py           # HTML report generation
  selective.py        # selective prediction and abstention utilities
  tta.py              # test-time augmentation uncertainty utilities
  visualization.py    # original previews, overlays, and side-by-side panels
scripts/
  benchmark_models.py       # multi-model benchmark runner
  create_study_packet.py    # full packet script wrapper
  dicom_to_png.py           # DICOM conversion/redaction wrapper
  evaluate_folder.py        # labeled-folder benchmarking with reliability metrics
  make_model_card.py        # markdown model-card generator
  predict_image.py          # simple prediction script wrapper
  selective_prediction.py   # selective prediction / abstention artifact generator
  tta_predict.py            # single-image TTA uncertainty wrapper
  tune_thresholds.py        # per-label threshold tuning
src/
  legacy Gradio and ensemble code from the original prototype
app.py                   # modern Gradio demo
docs/
  api.md                         # hosted API, DICOM, Docker, and audit workflow
  V0_3_RELIABILITY.md            # reliability workflow
  V0_4_REPORTING.md              # reporting workflow
  V0_6_RESEARCH_EVAL.md          # multi-model research evaluation workflow
  V0_7_SELECTIVE_PREDICTION.md   # selective prediction and abstention workflow
```

---

## Current limitations

- Predictions rely on pretrained TorchXRayVision models and should not be interpreted as clinically validated outputs for a new deployment setting.
- Thresholds tuned on one dataset should not be reused on another dataset without validation.
- Heatmaps and overlays show model sensitivity, not confirmed disease location.
- Evaluation currently assumes image-level binary labels, not radiologist-validated localization masks.
- Subgroup results can be unstable when group sizes are small or labels are imbalanced.
- TTA standard deviation is a rough uncertainty proxy and not a calibrated clinical confidence estimate.
- Selective prediction currently uses distance from the 0.5 decision boundary as a simple confidence score.
- DICOM redaction is a convenience helper, not a complete HIPAA de-identification pipeline.
- The hosted API is suitable for demos and internal research, not production clinical deployment.

---

## Concrete roadmap

### v0.8: stronger ML uncertainty layer

- Add ensemble variance across multiple TorchXRayVision models.
- Add TTA standard deviation as a selectable abstention score.
- Add conformal prediction sets for label-level uncertainty.
- Add subgroup-specific selective-risk curves.
- Add calibration transfer experiments across NIH, CheXpert, MIMIC-CXR, and PadChest where licenses permit.

### v0.9: clinical workflow prototype

- Add case queue, reviewer notes, and human-in-the-loop feedback capture.
- Add report comparison against radiology text labels where available.
- Add monitoring dashboard for drift, calibration, deferral rate, and alert volume.
- Add stronger deployment security boundaries and signed audit manifests.

---

## Responsible use

XRayMind should be framed as a research and education tool. For any real-world medical use, the project would need dataset governance, clinical validation, security review, deployment monitoring, regulatory review, and human oversight.
