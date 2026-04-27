# v0.7 Selective Prediction and Abstention

This workflow adds a research-only deferral layer to XRayMind. Instead of forcing the model to make an automatic prediction for every case, selective prediction evaluates what happens when the least-confident cases are deferred for human review.

> XRayMind is a research prototype only. It is not a medical device and is not intended for diagnosis, treatment, triage, or clinical decision-making.

## Why this matters

A chest X-ray classifier should not only report a probability. For safety-oriented research, it should also expose when the model is uncertain and when a case should be escalated. Selective prediction gives a simple way to study the tradeoff between:

- **Coverage:** fraction of cases automatically predicted.
- **Deferral rate:** fraction of cases sent to human review.
- **Selective risk:** error rate among the automatically predicted cases.
- **Selective accuracy:** accuracy among the automatically predicted cases.

This is useful for reliability research, demo narratives, model comparison, and human-in-the-loop product design.

## Confidence definition

For a binary pathology score `p`, XRayMind currently uses distance from the decision boundary:

```text
confidence = max(p, 1 - p)
```

Scores near 0 or 1 are treated as higher confidence. Scores near 0.5 are treated as lower confidence. This is simple and interpretable, but it is not a clinically validated uncertainty estimate.

## Generate selective artifacts from existing predictions

First run a benchmark or evaluation that writes predictions:

```bash
python scripts/benchmark_models.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all \
  --out-dir outputs/benchmark_v0_7
```

Then generate abstention artifacts:

```bash
python scripts/selective_prediction.py \
  --labels data/labels.csv \
  --predictions outputs/benchmark_v0_7/densenet121-res224-all/predictions.csv \
  --model densenet121-res224-all \
  --dataset "XRayMind folder dataset" \
  --out-dir outputs/selective_v0_7 \
  --max-risk 0.15
```

Outputs:

```text
outputs/selective_v0_7/selective_curves.csv
outputs/selective_v0_7/selective_summary.csv
outputs/selective_v0_7/operating_point.json
outputs/selective_v0_7/selective_risk_curve.png
outputs/selective_v0_7/SELECTIVE_PREDICTION_REPORT.md
```

## Generate selective artifacts during a benchmark

```bash
python scripts/benchmark_models.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all densenet121-res224-chex \
  --out-dir outputs/benchmark_v0_7 \
  --selective \
  --max-risk 0.15 \
  --min-coverage 0.6
```

This adds:

```text
outputs/benchmark_v0_7/combined_selective_summary.csv
outputs/benchmark_v0_7/<model>/selective_prediction/selective_curves.csv
outputs/benchmark_v0_7/<model>/selective_prediction/selective_summary.csv
outputs/benchmark_v0_7/<model>/selective_prediction/operating_point.json
outputs/benchmark_v0_7/<model>/selective_prediction/selective_risk_curve.png
outputs/benchmark_v0_7/<model>/selective_prediction/SELECTIVE_PREDICTION_REPORT.md
```

## Recommended interpretation

A useful model should ideally maintain lower selective risk as coverage decreases. If selective risk does not improve when uncertain cases are deferred, then the confidence heuristic is not sorting easy and hard cases well.

For a paper or product demo, report at least three operating points:

- 100% coverage: no deferral baseline.
- 80% coverage: moderate deferral.
- 50% coverage: high-deferral, safety-oriented mode.

Also report the chosen operating point from `operating_point.json`, especially when using `--max-risk` or `--min-coverage`.

## Example demo narrative

Instead of saying “the model predicts every X-ray,” say:

> XRayMind can be configured as a human-in-the-loop research system. It automatically predicts high-confidence cases and defers uncertain cases for expert review. The selective-risk curve shows how much error drops as coverage is reduced.

## Limitations

- The confidence score is based only on probability distance from 0.5.
- It does not guarantee that deferred cases are clinically harder.
- It does not replace proper uncertainty estimation, calibration, prospective validation, or clinical workflow testing.
- It assumes binary image-level labels.
- For imbalanced labels, accuracy and selective risk can be misleading unless reported with sensitivity, specificity, and prevalence.

## Next upgrade ideas

- Add ensemble-based uncertainty using multiple TorchXRayVision checkpoints.
- Add TTA standard deviation as an uncertainty score.
- Add conformal prediction sets for label-level abstention.
- Add subgroup-specific deferral curves.
- Add a reviewer queue UI for deferred cases.
