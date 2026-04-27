# v0.8 Ensemble Uncertainty Workflow

XRayMind v0.8 adds a multi-model uncertainty workflow for research benchmarking. Instead of relying on one pretrained TorchXRayVision checkpoint, the new workflow runs several model variants, averages their probabilities, and records disagreement signals that can be used for abstention and human review routing.

> XRayMind is not a medical device. Ensemble uncertainty is a research signal, not a calibrated clinical confidence score.

---

## Why this matters

Single-model probabilities can look overconfident on shifted hospital data, different scanner protocols, pediatric cohorts, portable AP images, or labels created with different radiology-report rules. A simple ensemble gives the tool a stronger reliability story:

- the prediction is the mean probability across model members;
- disagreement is exposed through standard deviation, range, entropy, and a combined uncertainty score;
- selective prediction can defer cases with high ensemble uncertainty;
- every run writes reproducible CSV outputs, reports, and a run manifest.

---

## Expected label CSV

```csv
image,Atelectasis,Cardiomegaly,Effusion,sex,view_position
000001.png,0,1,0,F,PA
000002.png,1,0,1,M,AP
```

The `image` column is joined against files in `--image-dir`. All non-image and non-subgroup columns are treated as binary pathology labels unless `--labels-to-evaluate` is provided.

---

## Run ensemble prediction

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

This produces:

```text
outputs/ensemble_v0_8/ensemble_predictions.csv
outputs/ensemble_v0_8/ensemble_metrics.csv
outputs/ensemble_v0_8/ensemble_uncertainty_summary.csv
outputs/ensemble_v0_8/ENSEMBLE_UNCERTAINTY_REPORT.md
outputs/ensemble_v0_8/run_manifest.json
outputs/ensemble_v0_8/members/<model>_predictions.csv
outputs/ensemble_v0_8/selective/selective_curves.csv
outputs/ensemble_v0_8/selective/selective_summary.csv
outputs/ensemble_v0_8/selective/operating_point.json
outputs/ensemble_v0_8/selective/selective_risk_curve.png
outputs/ensemble_v0_8/selective/SELECTIVE_PREDICTION_REPORT.md
```

---

## Ensemble prediction CSV columns

For each pathology label, the ensemble CSV keeps the standard XRayMind prediction format while adding uncertainty columns:

```text
Cardiomegaly                       # ensemble mean probability
Cardiomegaly_ensemble_std          # standard deviation across model probabilities
Cardiomegaly_ensemble_range        # max minus min probability across models
Cardiomegaly_ensemble_entropy      # entropy of ensemble mean probability
Cardiomegaly_ensemble_uncertainty  # selected uncertainty score
Cardiomegaly_confidence            # confidence ranking used for selective prediction
Cardiomegaly__<model-name>         # raw probability from each member model
```

Existing metric utilities continue to read the plain label column, so `ensemble_predictions.csv` can be passed into the normal evaluation and selective prediction scripts.

---

## Uncertainty method options

```bash
--uncertainty-method std
--uncertainty-method range
--uncertainty-method entropy
--uncertainty-method std_entropy
```

Recommended starting point: `std`. It directly captures model disagreement and is easy to explain.

Use `std_entropy` when you want uncertainty to increase both when models disagree and when the ensemble mean is close to the binary decision boundary.

---

## Run selective prediction from ensemble outputs

You can rerun abstention analysis from a saved ensemble CSV:

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

---

## How to present this in the repo or demo

A strong demo narrative is:

1. Run one image or a small folder through several pretrained chest X-ray classifiers.
2. Show the ensemble mean predictions.
3. Show model disagreement for each label.
4. Route high-uncertainty cases to review instead of pretending the model is always confident.
5. Show the risk-coverage curve to explain the safety/reliability tradeoff.

This makes XRayMind look less like a basic chest X-ray classifier and more like a responsible AI reliability toolkit for medical-imaging research.

---

## Limitations

- Ensemble disagreement is not the same as true clinical uncertainty.
- The model family may share training data or architecture biases.
- Label definitions differ across NIH, CheXpert, MIMIC-CXR, PadChest, and local datasets.
- Uncertainty needs external validation before any clinical workflow claims.
- Deferral improves safety only if deferred cases receive qualified human review.
