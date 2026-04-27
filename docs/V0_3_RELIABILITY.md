# XRayMind v0.3 Reliability Workflow

This workflow upgrades XRayMind from a demo into a more serious evaluation toolkit. It adds threshold tuning, calibration metrics, bootstrap confidence intervals, reliability plots, and model-card generation.

> XRayMind remains a research prototype only. It is not for clinical diagnosis or treatment decisions.

## 1. Prepare labels

Use a CSV with one image filename column and one binary column per label.

```csv
image,Atelectasis,Cardiomegaly,Effusion
000001.png,0,1,0
000002.png,1,0,1
```

## 2. Run evaluation

```bash
python scripts/evaluate_folder.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --image-column image \
  --out outputs/eval.csv \
  --predictions-out outputs/predictions.csv \
  --tune-thresholds \
  --threshold-objective f1 \
  --bootstrap 1000 \
  --save-plots
```

Outputs:

```text
outputs/eval.csv
outputs/predictions.csv
outputs/reliability_plots/*_reliability.png
```

## 3. Tune thresholds separately

If predictions already exist, tune thresholds without re-running model inference:

```bash
python scripts/tune_thresholds.py \
  --labels data/labels.csv \
  --predictions outputs/predictions.csv \
  --out outputs/thresholds.csv \
  --objective f1
```

Supported objectives:

- `f1`: balances precision and recall.
- `youden`: maximizes sensitivity + specificity - 1.

## 4. Generate a model card

```bash
python scripts/make_model_card.py \
  --metrics outputs/eval.csv \
  --model densenet121-res224-all \
  --dataset "NIH ChestX-ray14 validation split" \
  --out outputs/MODEL_CARD.md
```

## 5. Metrics included

- AUROC
- AUPRC
- Brier score
- Expected calibration error
- Sensitivity
- Specificity
- Precision
- F1
- Per-label threshold
- Optional AUROC/AUPRC bootstrap confidence intervals
- Optional reliability diagram per label

## 6. How to interpret results

A strong X-ray model should not only have high AUROC. It should also be reasonably calibrated and stable across labels and datasets. In practice:

- High AUROC but poor ECE means ranking is useful but probabilities are unreliable.
- High sensitivity but low specificity means the model may over-alert.
- High specificity but low sensitivity means the model may miss positives.
- Thresholds tuned on one dataset should not be reused on another dataset without validation.
- Reliability diagrams can reveal systematic overconfidence or underconfidence.

## 7. Next reliability improvements

- Add train/validation/test split helpers.
- Add subgroup evaluation when metadata is available.
- Add confidence intervals for F1, sensitivity, and specificity.
- Add temperature scaling or isotonic calibration.
- Add multi-model comparison tables.
