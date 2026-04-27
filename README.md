# XRayMind

XRayMind is an explainable chest X-ray research prototype built around TorchXRayVision models, Captum attribution methods, reliability evaluation, reporting, DICOM ingestion, hosted API inference, uncertainty, model cards, dataset cards, selective prediction, ensemble disagreement, conformal prediction sets, local case review workflows, and a lightweight Gradio interface. The project is intended for AI/ML research demos, model inspection, benchmarking, and educational exploration of chest radiography classifiers.

> **Important:** XRayMind is not a medical device and is not for clinical diagnosis, treatment, or triage. Outputs must be reviewed by qualified clinical professionals.

---

## What is new in v1.0

- Local SQLite case workflow in `xraymind/store.py` and `xraymind/cases.py`.
- Case queue records with image path, model, priority, status, tags, predictions, reviews, and timestamps.
- Human review capture with decisions, reviewer notes, final labels, and audit events.
- Dashboard summaries in `xraymind/dashboard.py` for status counts, priorities, low-confidence cases, review decisions, and top alert labels.
- FastAPI workflow routes: `/cases`, `/cases/{case_id}`, `/cases/{case_id}/review`, `/dashboard/summary`, and `/dashboard/attention`.
- CLI workflow commands: `xraymind case create/list/show/status/review` and `xraymind dashboard`.
- Runnable local demo script: `scripts/case_workflow_demo.py`.
- Dedicated workflow guide in `docs/V1_0_CASE_WORKFLOW.md`.

## What changed in v0.9

- Split-conformal prediction-set utilities in `xraymind/conformal.py`.
- Standalone conformal runner: `scripts/conformal_predict.py`.
- Per-label conformal thresholds with target coverage control.
- Prediction-set outputs for `negative`, `positive`, ambiguous, and empty sets.
- Empirical coverage, mean set size, singleton rate, ambiguous rate, and empty rate summaries.
- Markdown conformal reports for safer uncertainty communication.

## What changed in v0.8

- Multi-model ensemble prediction utilities in `xraymind/ensemble.py`.
- Standalone ensemble runner: `scripts/ensemble_predict.py`.
- Ensemble mean probabilities with per-label disagreement signals.
- Uncertainty columns for standard deviation, range, entropy, and selected uncertainty score.
- Ensemble-aware selective prediction using uncertainty-based deferral.
- Per-member prediction CSVs, ensemble metrics, uncertainty summaries, Markdown reports, and run manifests.
- Dedicated workflow guide in `docs/V0_8_ENSEMBLE_UNCERTAINTY.md`.

## What changed in v0.7

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

### Create a local review case

```bash
xraymind case create \
  --image path/to/chest_xray.png \
  --priority elevated \
  --tags demo,review \
  --db outputs/xraymind_cases.sqlite3
```

Add a reviewer decision:

```bash
xraymind case review 1 \
  --decision uncertain \
  --reviewer reviewer_a \
  --notes "Needs second read because model confidence is low."
```

Show the dashboard:

```bash
xraymind dashboard
xraymind dashboard --attention --limit 20
```

See `docs/V1_0_CASE_WORKFLOW.md` for the complete local and FastAPI workflow.

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

Optional PDF:

```bash
xraymind packet \
  --image path/to/chest_xray.png \
  --out-dir outputs/study_packet \
  --pdf
```

---

## Hosted API workflow

Start the API:

```bash
uvicorn xraymind.api:app --reload
```

Create a review case through the API:

```bash
curl -X POST http://localhost:8000/cases \
  -F "file=@path/to/chest_xray.png" \
  -F "priority=elevated" \
  -F "tags=demo,review"
```

Add a review:

```bash
curl -X POST http://localhost:8000/cases/1/review \
  -F "decision=uncertain" \
  -F "reviewer=reviewer_a" \
  -F "notes=Needs second read"
```

View dashboard:

```bash
curl http://localhost:8000/dashboard/summary
curl http://localhost:8000/dashboard/attention?limit=20
```

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

### Run ensemble uncertainty evaluation

```bash
python scripts/ensemble_predict.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all densenet121-res224-nih densenet121-res224-chex \
  --out-dir outputs/ensemble_v0_8 \
  --dataset "XRayMind folder dataset" \
  --selective \
  --max-risk 0.15
```

This produces ensemble mean predictions, per-model prediction CSVs, uncertainty summaries, ensemble metrics, a Markdown ensemble report, and optional uncertainty-aware selective prediction artifacts.

### Run conformal prediction sets from existing predictions

```bash
python scripts/conformal_predict.py \
  --labels data/labels.csv \
  --predictions outputs/ensemble_v0_8/ensemble_predictions.csv \
  --out-dir outputs/conformal_v0_9 \
  --alpha 0.1 \
  --calibration-fraction 0.5 \
  --dataset-name "XRayMind folder dataset"
```

This produces:

```text
outputs/conformal_v0_9/conformal_thresholds.csv
outputs/conformal_v0_9/conformal_predictions.csv
outputs/conformal_v0_9/conformal_summary.csv
outputs/conformal_v0_9/CONFORMAL_REPORT.md
```

Use conformal sets as a calibration-backed uncertainty layer: singleton sets are more decisive, ambiguous sets should be reviewed, and coverage depends on the calibration split matching the evaluation distribution.

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

For ensemble uncertainty outputs:

```bash
python scripts/selective_prediction.py \
  --labels data/labels.csv \
  --predictions outputs/ensemble_v0_8/ensemble_predictions.csv \
  --model ensemble-xraymind \
  --dataset "XRayMind folder dataset" \
  --out-dir outputs/ensemble_selective_v0_8 \
  --confidence-method ensemble_uncertainty \
  --uncertainty-suffix _ensemble_uncertainty \
  --max-risk 0.15
```

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

See `docs/V0_3_RELIABILITY.md`, `docs/V0_6_RESEARCH_EVAL.md`, `docs/V0_7_SELECTIVE_PREDICTION.md`, `docs/V0_8_ENSEMBLE_UNCERTAINTY.md`, `docs/V0_9_CONFORMAL_PREDICTION.md`, and `docs/V1_0_CASE_WORKFLOW.md` for the full reliability, benchmark, abstention, ensemble uncertainty, conformal prediction, and case review workflows.

---

## Project structure

```text
xraymind/
  api.py              # FastAPI hosted inference and case workflow service
  audit.py            # JSONL audit logging helpers
  bootstrap.py        # bootstrap confidence intervals
  cases.py            # local case queue, prediction, and review helpers
  cli.py              # command-line interface
  conformal.py        # conformal prediction-set utilities
  config.py           # shared constants and safety disclaimer
  dashboard.py        # case workflow dashboard summaries
  dicom.py            # DICOM ingestion, conversion, and redaction helpers
  ensemble.py         # multi-model ensemble prediction and uncertainty utilities
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
  store.py            # SQLite persistence for cases, predictions, reviews, audit events
  tta.py              # test-time augmentation uncertainty utilities
  visualization.py    # original previews, overlays, and side-by-side panels
scripts/
  benchmark_models.py       # multi-model benchmark runner
  case_workflow_demo.py     # local case/review/dashboard demo
  conformal_predict.py      # conformal prediction-set runner
  create_study_packet.py    # full packet script wrapper
  dicom_to_png.py           # DICOM conversion/redaction wrapper
  ensemble_predict.py       # multi-model ensemble uncertainty runner
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
  V0_8_ENSEMBLE_UNCERTAINTY.md   # ensemble uncertainty workflow
  V0_9_CONFORMAL_PREDICTION.md   # conformal prediction-set workflow
  V1_0_CASE_WORKFLOW.md          # local case workflow and dashboard guide
```

---

## Current limitations

- Predictions rely on pretrained TorchXRayVision models and should not be interpreted as clinically validated outputs for a new deployment setting.
- Thresholds tuned on one dataset should not be reused on another dataset without validation.
- Heatmaps and overlays show model sensitivity, not confirmed disease location.
- Evaluation currently assumes image-level binary labels, not radiologist-validated localization masks.
- Subgroup results can be unstable when group sizes are small or labels are imbalanced.
- TTA standard deviation and ensemble disagreement are rough uncertainty proxies, not calibrated clinical confidence estimates.
- Selective prediction can now use either probability-margin confidence or ensemble uncertainty, but deferral only improves safety if deferred cases receive qualified review.
- Conformal coverage depends on exchangeability between calibration and evaluation data, so distribution shift can break the coverage guarantee.
- The v1.0 case workflow is a local research prototype, not an EHR/RIS/PACS integration.
- DICOM redaction is a convenience helper, not a complete HIPAA de-identification pipeline.
- The hosted API is suitable for demos and internal research, not production clinical deployment.

---

## Concrete roadmap

### v1.1: case workflow UI and monitoring

- Add a Gradio case queue tab for pending, deferred, flagged, and reviewed cases.
- Export case-review datasets to CSV/JSONL for model improvement and error analysis.
- Add reviewer agreement analytics and disagreement galleries.
- Add drift monitoring over prediction distributions, alert volume, deferral rate, and label prevalence.
- Add signed audit manifests for packet and case exports.

### v1.2: stronger validation and generalization

- Add subgroup-specific conformal and selective-risk curves.
- Add calibration transfer experiments across NIH, CheXpert, MIMIC-CXR, and PadChest where licenses permit.
- Add TTA-plus-ensemble hybrid uncertainty scoring.
- Add uncertainty calibration plots and failure-case galleries.
- Add optional Postgres backend if deploying beyond local demos.

---

## Responsible use

XRayMind should be framed as a research and education tool. For any real-world medical use, the project would need dataset governance, clinical validation, security review, deployment monitoring, regulatory review, and human oversight.
