# XRayMind

XRayMind is an explainable chest X-ray research prototype built around TorchXRayVision models, Captum attribution methods, reliability evaluation, reporting, DICOM ingestion, hosted API inference, uncertainty, model cards, dataset cards, selective prediction, ensemble disagreement, conformal prediction sets, local case review workflows, exportable review datasets, operational monitoring snapshots, assignment queues, second-reader review, and a lightweight Gradio interface. The project is intended for AI/ML research demos, model inspection, benchmarking, and educational exploration of chest radiography classifiers.

> **Important:** XRayMind is not a medical device and is not for clinical diagnosis, treatment, or triage. Outputs must be reviewed by qualified clinical professionals.

---

## What is new in v1.2

- Assignment fields for review cases: `assigned_to`, `due_at`, and `needs_second_reader`.
- Idempotent SQLite schema migrations so existing local workflow databases are upgraded safely.
- Second-reader logic: uncertain, disagree, defer, and flag first-round reviews automatically mark the case for another read.
- New CLI command: `xraymind case assign`.
- New API route: `POST /cases/{case_id}/assign`.
- Case list filters for `assigned_to` and `needs_second_reader`.
- Dashboard metrics for assignment load, unassigned cases, and second-reader cases.
- Case exports now include assignment and second-reader columns.

## What is new in v1.1

- Case export utilities in `xraymind/export.py` for JSONL/CSV review datasets.
- Export manifests with file paths, row counts, filters, SHA-256 checksums, and responsible-use disclaimer.
- Monitoring snapshots in `xraymind/monitoring.py` for review rate, low-confidence rate, disagreement rate, flagged/deferred/urgent rates, status counts, decision counts, and top alert labels.
- Simple drift comparison against a previous monitoring snapshot.
- Standalone scripts: `scripts/export_cases.py` and `scripts/monitor_cases.py`.
- Tests for export integrity, reviewer disagreement analytics, and drift alerts.

## What is new in v1.0

- Local SQLite case workflow in `xraymind/store.py` and `xraymind/cases.py`.
- Case queue records with image path, model, priority, status, tags, predictions, reviews, and timestamps.
- Human review capture with decisions, reviewer notes, final labels, and audit events.
- Dashboard summaries in `xraymind/dashboard.py` for status counts, priorities, low-confidence cases, review decisions, and top alert labels.
- FastAPI workflow routes: `/cases`, `/cases/{case_id}`, `/cases/{case_id}/review`, `/dashboard/summary`, and `/dashboard/attention`.
- CLI workflow commands: `xraymind case create/list/show/status/review` and `xraymind dashboard`.
- Runnable local demo script: `scripts/case_workflow_demo.py`.
- Dedicated workflow guide in `docs/V1_0_CASE_WORKFLOW.md`.

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
  --assigned-to reviewer_a \
  --db outputs/xraymind_cases.sqlite3
```

Assign or reassign a case:

```bash
xraymind case assign 1 \
  --reviewer reviewer_b \
  --due-at 2026-05-01T17:00:00Z \
  --needs-second-reader
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

Export the review dataset:

```bash
python scripts/export_cases.py \
  --db outputs/xraymind_cases.sqlite3 \
  --out-dir outputs/exports
```

Create a monitoring snapshot:

```bash
python scripts/monitor_cases.py \
  --db outputs/xraymind_cases.sqlite3 \
  --out outputs/monitoring/snapshot.json
```

Compare against a previous snapshot for simple drift alerts:

```bash
python scripts/monitor_cases.py \
  --db outputs/xraymind_cases.sqlite3 \
  --baseline outputs/monitoring/baseline.json \
  --out outputs/monitoring/current.json
```

See `docs/V1_0_CASE_WORKFLOW.md` for the complete local and FastAPI workflow.

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
  -F "assigned_to=reviewer_a" \
  -F "tags=demo,review"
```

Assign a case:

```bash
curl -X POST http://localhost:8000/cases/1/assign \
  -F "reviewer=reviewer_b" \
  -F "due_at=2026-05-01T17:00:00Z" \
  -F "needs_second_reader=true"
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

---

## Project structure

```text
xraymind/
  api.py              # FastAPI hosted inference and case workflow service
  audit.py            # JSONL audit logging helpers
  bootstrap.py        # bootstrap confidence intervals
  cases.py            # local case queue, prediction, assignment, and review helpers
  cli.py              # command-line interface
  conformal.py        # conformal prediction-set utilities
  config.py           # shared constants and safety disclaimer
  dashboard.py        # case workflow dashboard summaries
  dicom.py            # DICOM ingestion, conversion, and redaction helpers
  ensemble.py         # multi-model ensemble prediction and uncertainty utilities
  evaluation.py       # reusable benchmark, dataset-card, and model-card helpers
  export.py           # case workflow CSV/JSONL export utilities
  explainability.py   # Captum attribution utilities
  inference.py        # structured prediction output
  metrics.py          # reliability metrics and threshold tuning
  model_loader.py     # cached TorchXRayVision model loading
  monitoring.py       # workflow monitoring and drift snapshots
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
  export_cases.py           # case workflow CSV/JSONL export runner
  make_model_card.py        # markdown model-card generator
  monitor_cases.py          # case workflow monitoring snapshot runner
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
- The case workflow, export, and monitoring layers are local research prototypes, not EHR/RIS/PACS integrations.
- Assignment and second-reader logic are workflow helpers only; they are not clinical triage rules.
- DICOM redaction is a convenience helper, not a complete HIPAA de-identification pipeline.
- The hosted API is suitable for demos and internal research, not production clinical deployment.

---

## Concrete roadmap

### v1.3: hosted productization layer

- Add authentication roles for admin, reviewer, and read-only auditor.
- Add tenant-aware hosted queues and Postgres persistence.
- Add asynchronous background processing for uploaded images.
- Add export redaction controls and audit-log signing.
- Add basic PACS-style import/export mocks for demo environments.

### v1.4: stronger validation and generalization

- Add subgroup-specific conformal and selective-risk curves.
- Add calibration transfer experiments across NIH, CheXpert, MIMIC-CXR, and PadChest where licenses permit.
- Add TTA-plus-ensemble hybrid uncertainty scoring.
- Add uncertainty calibration plots and failure-case galleries.
- Add dataset shift monitoring reports.

---

## Responsible use

XRayMind should be framed as a research and education tool. For any real-world medical use, the project would need dataset governance, clinical validation, security review, deployment monitoring, regulatory review, and human oversight.
