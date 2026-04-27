# XRayMind v0.9: Conformal Prediction Sets

XRayMind v0.9 adds a calibration-backed uncertainty layer using split conformal prediction. Instead of only returning a probability and a hard thresholded label, this workflow returns a small prediction set for each pathology label.

The goal is to make uncertainty easier to communicate in research demos and evaluation reports. A conformal set can contain:

- `negative`: the model has enough calibrated evidence for the negative class.
- `positive`: the model has enough calibrated evidence for the positive class.
- `negative|positive`: the case is ambiguous and should be reviewed.
- `empty`: the calibrated threshold excludes both classes, which is a warning sign that the case is outside the model's reliable operating region.

> XRayMind is not a medical device. Conformal outputs are research reliability signals and should not be used for clinical diagnosis, triage, or treatment.

---

## Why this matters

A chest X-ray tool should not only say what it predicts. It should also say when it is uncertain. Earlier XRayMind versions added reliability metrics, selective prediction, and ensemble disagreement. Conformal prediction adds a complementary layer: a target coverage level controlled by `alpha`.

For example, `alpha=0.1` targets 90 percent marginal coverage under the calibration/evaluation distribution assumptions. This does not mean the model is clinically safe. It means that, if the calibration and evaluation samples are exchangeable and labels are reliable, the prediction sets should include the true binary class at roughly the target rate.

---

## Run conformal prediction from existing predictions

First produce prediction probabilities with one of the existing workflows:

```bash
python scripts/ensemble_predict.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all densenet121-res224-nih densenet121-res224-chex \
  --out-dir outputs/ensemble_v0_8 \
  --dataset "XRayMind folder dataset"
```

Then run conformal prediction:

```bash
python scripts/conformal_predict.py \
  --labels data/labels.csv \
  --predictions outputs/ensemble_v0_8/ensemble_predictions.csv \
  --out-dir outputs/conformal_v0_9 \
  --alpha 0.1 \
  --calibration-fraction 0.5 \
  --dataset-name "XRayMind folder dataset"
```

The script creates:

```text
outputs/conformal_v0_9/conformal_thresholds.csv
outputs/conformal_v0_9/conformal_predictions.csv
outputs/conformal_v0_9/conformal_summary.csv
outputs/conformal_v0_9/CONFORMAL_REPORT.md
```

---

## Output files

### `conformal_thresholds.csv`

One row per label:

- `label`: pathology label.
- `alpha`: requested miscoverage rate.
- `target_coverage`: `1 - alpha`.
- `qhat`: finite-sample conformal threshold.
- `n_calibration`: number of calibration examples used for that label.
- `mean_nonconformity`: average calibration nonconformity score.
- `max_nonconformity`: largest calibration nonconformity score.

### `conformal_predictions.csv`

This contains the original prediction probabilities plus columns such as:

```text
Cardiomegaly_conformal_set
Cardiomegaly_conformal_set_size
Cardiomegaly_conformal_includes_negative
Cardiomegaly_conformal_includes_positive
Cardiomegaly_conformal_qhat
```

### `conformal_summary.csv`

One row per label:

- `empirical_coverage`: fraction of evaluation cases where the set contains the true class.
- `coverage_gap`: empirical coverage minus target coverage.
- `mean_set_size`: average prediction-set size.
- `singleton_rate`: fraction of one-label sets.
- `ambiguous_rate`: fraction of two-label sets.
- `empty_rate`: fraction of empty sets.

### `CONFORMAL_REPORT.md`

A compact Markdown report for demos, internal evaluation, or project documentation.

---

## How to interpret the sets

Use conformal sets as a review-routing layer rather than a diagnosis layer.

A singleton `positive` set means the calibrated set only includes the positive class. A singleton `negative` set means the calibrated set only includes the negative class. A `negative|positive` set means the prediction is not decisive at the selected coverage target. An `empty` set should be treated carefully because it suggests a mismatch between the probability and the calibrated threshold behavior.

For a safer demo, show the prediction probability, ensemble uncertainty or selective deferral flag, and conformal set together. That makes it clear that the tool is a research reliability prototype rather than a clinical decision system.

---

## Limitations

Conformal prediction depends on assumptions. The most important one is that calibration and evaluation samples should come from the same distribution. If a model is calibrated on one hospital, scanner type, patient population, or labeling protocol and evaluated on a different one, empirical coverage may fail.

Conformal prediction also inherits label noise. Public chest X-ray datasets often use automatically extracted report labels, so the conformal guarantee is only with respect to those labels, not necessarily ground-truth clinical findings.

---

## Recommended next experiments

1. Compare conformal set size across TorchXRayVision models and ensemble outputs.
2. Report empirical coverage by subgroup, such as sex, view position, hospital site, or scanner source.
3. Measure how ambiguous-rate changes under distribution shift.
4. Combine conformal sets with selective prediction: automatically handle singleton low-risk cases and defer ambiguous/high-uncertainty cases.
5. Build a failure gallery of high-confidence wrong singleton sets and empty sets.
