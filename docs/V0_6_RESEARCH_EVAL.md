# v0.6 Research Evaluation Workflow

This release adds a research-grade evaluation path for comparing multiple pretrained chest X-ray classifiers on folder/CSV datasets.

> XRayMind is a research prototype only. It is not a medical device and is not intended for diagnosis, treatment, triage, or clinical decision-making.

## Dataset format

Use one CSV row per image. The default image filename column is `image`. Every other binary column is treated as a pathology label unless you pass subgroup columns.

```csv
image,Atelectasis,Cardiomegaly,Effusion,sex,view_position
000001.png,0,1,0,F,PA
000002.png,1,0,1,M,AP
```

Folder layout:

```text
data/images/000001.png
data/images/000002.png
data/labels.csv
```

DICOM files can be referenced in the image column if `pydicom` is installed.

## Multi-model benchmark

```bash
python scripts/benchmark_models.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all densenet121-res224-nih densenet121-res224-chex \
  --out-dir outputs/benchmark_v0_6 \
  --save-plots
```

Outputs:

```text
outputs/benchmark_v0_6/DATASET_CARD.md
outputs/benchmark_v0_6/leaderboard.csv
outputs/benchmark_v0_6/combined_metrics.csv
outputs/benchmark_v0_6/run_manifest.json
outputs/benchmark_v0_6/<model>/predictions.csv
outputs/benchmark_v0_6/<model>/metrics.csv
outputs/benchmark_v0_6/<model>/reliability_plots/*.png
```

## Subgroup evaluation

If your labels CSV includes metadata such as sex, age group, scanner, site, or view position, pass those columns as subgroups:

```bash
python scripts/benchmark_models.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all densenet121-res224-chex \
  --subgroups sex view_position site \
  --min-group-size 25 \
  --out-dir outputs/benchmark_subgroups
```

This writes:

```text
outputs/benchmark_subgroups/combined_subgroup_metrics.csv
outputs/benchmark_subgroups/<model>/subgroup_metrics.csv
```

Use this to inspect robustness gaps across metadata slices. Small subgroups are skipped by `--min-group-size`.

## Threshold tuning

By default, metrics use a fixed threshold of `0.5`. You can tune thresholds per label using either F1 or Youden objective:

```bash
python scripts/benchmark_models.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all \
  --tune-thresholds \
  --threshold-objective f1
```

For a clean paper-style study, tune thresholds on a validation split, then report final numbers on a held-out test split.

## Bootstrap confidence intervals

```bash
python scripts/benchmark_models.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all \
  --bootstrap 1000
```

This adds AUROC and AUPRC confidence interval columns to each model's `metrics.csv` and the combined metrics table.

## Test-time augmentation uncertainty

For a single case:

```bash
python scripts/tta_predict.py \
  --image path/to/chest_xray.png \
  --model densenet121-res224-all \
  --out outputs/tta_prediction.json
```

The result includes mean and standard deviation across simple augmentations. Treat this as a rough research signal, not calibrated clinical uncertainty.

## Recommended v0.6 experiment set

1. Run a small smoke benchmark with `--limit 20`.
2. Run all selected models on the full dataset.
3. Enable `--save-plots` for reliability diagrams.
4. Add subgroup columns if metadata is available.
5. Generate model cards from the resulting metrics.
6. Compare models using both discrimination metrics and calibration metrics.

## Reporting guidance

For papers or demos, do not report only AUROC. Include:

- AUROC and AUPRC for discrimination.
- Brier score and ECE for reliability.
- Sensitivity, specificity, precision, and F1 at a stated threshold.
- Confidence intervals when sample size permits.
- Subgroup results when metadata is available.
- Clear statement that heatmaps are model-sensitivity visualizations, not clinical localization evidence.
