# XRayMind v1.0 Case Workflow

XRayMind v1.0 adds a lightweight local review workflow around the existing chest X-ray prediction, explanation, reporting, uncertainty, ensemble, and conformal utilities.

The workflow is intentionally simple: it stores cases in SQLite, keeps the original inference path unchanged, captures human review decisions, and exposes dashboard summaries for model monitoring demos.

> XRayMind is not a medical device and is not for clinical diagnosis, treatment, triage, or deployment without proper clinical, regulatory, privacy, and security review.

---

## What this adds

- Local SQLite case store.
- Case records with image path, model name, priority, status, tags, and timestamps.
- Persisted prediction payloads linked to cases.
- Human reviewer notes and decisions.
- Audit events for case creation, prediction creation, status updates, and review creation.
- FastAPI routes for case queue operations.
- CLI commands for local workflow demos.
- Dashboard summary and attention queue helpers.

---

## Data model

The default database path is:

```text
outputs/xraymind_cases.sqlite3
```

Tables:

```text
metadata
cases
predictions
reviews
audit_events
```

Case statuses:

```text
pending, reviewed, deferred, flagged, archived
```

Priorities:

```text
routine, elevated, urgent
```

Review decisions:

```text
agree, disagree, uncertain, defer, flag
```

---

## CLI workflow

Create a case and run prediction:

```bash
xraymind case create \
  --image path/to/chest_xray.png \
  --priority elevated \
  --tags demo,validation \
  --db outputs/xraymind_cases.sqlite3
```

List cases:

```bash
xraymind case list --status pending --limit 10
```

Show one case with latest prediction and review history:

```bash
xraymind case show 1
```

Add a human review:

```bash
xraymind case review 1 \
  --decision uncertain \
  --reviewer reviewer_a \
  --notes "Needs second read because model confidence is low."
```

Update status directly:

```bash
xraymind case status 1 flagged
```

Show dashboard summary:

```bash
xraymind dashboard
```

Show attention queue:

```bash
xraymind dashboard --attention --limit 20
```

Run the full demo script:

```bash
python scripts/case_workflow_demo.py \
  --image path/to/chest_xray.png \
  --decision uncertain
```

---

## API workflow

Start the API:

```bash
uvicorn xraymind.api:app --reload
```

Optional API key:

```bash
export XRAYMIND_API_KEY=replace-me
```

Optional workflow DB location:

```bash
export XRAYMIND_DB_PATH=outputs/xraymind_cases.sqlite3
```

Optional uploaded case image directory:

```bash
export XRAYMIND_CASE_UPLOAD_DIR=outputs/case_uploads
```

### Create a case

```bash
curl -X POST http://localhost:8000/cases \
  -F "file=@path/to/chest_xray.png" \
  -F "priority=elevated" \
  -F "tags=demo,review" \
  -F "top_k=5"
```

With API key:

```bash
curl -X POST http://localhost:8000/cases \
  -H "X-API-Key: replace-me" \
  -F "file=@path/to/chest_xray.png"
```

### List and inspect cases

```bash
curl "http://localhost:8000/cases?status=pending&limit=10"
curl "http://localhost:8000/cases/1"
```

### Add review

```bash
curl -X POST http://localhost:8000/cases/1/review \
  -F "decision=uncertain" \
  -F "reviewer=reviewer_a" \
  -F "notes=Needs second read" \
  -F 'final_labels_json={"Cardiomegaly":"uncertain"}'
```

### Dashboard

```bash
curl http://localhost:8000/dashboard/summary
curl http://localhost:8000/dashboard/attention?limit=20
```

---

## Dashboard fields

`xraymind.dashboard.dashboard_summary()` returns:

```text
total_cases
pending_cases
reviewed_cases
deferred_cases
flagged_cases
low_confidence_cases
total_reviews
reviewer_disagreement_rate
status_counts
priority_counts
review_decision_counts
top_alert_labels
```

`xraymind.dashboard.cases_requiring_attention()` prioritizes:

1. urgent cases,
2. flagged or deferred cases,
3. pending cases,
4. low-confidence model outputs.

---

## How to use this in a demo

A strong demo narrative is:

1. Upload a chest X-ray.
2. XRayMind creates a case and runs prediction.
3. The dashboard shows pending/low-confidence cases.
4. A reviewer adds a decision and notes.
5. The case status changes from `pending` to `reviewed`, `deferred`, or `flagged`.
6. The dashboard updates review and alert counts.

This turns XRayMind from a single-image classifier into a review workflow prototype.

---

## Next improvements

Recommended next v1.1 work:

- Add a simple Gradio case queue tab.
- Export case-review datasets to CSV/JSONL for model improvement.
- Add reviewer agreement analytics.
- Add drift monitoring over prediction distributions and label prevalence.
- Add signed audit manifests for packet and case exports.
- Add optional Postgres backend if deploying beyond local demos.
