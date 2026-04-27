"""Run a multi-model XRayMind benchmark on a folder/CSV dataset.

Example:
python scripts/benchmark_models.py \
  --image-dir data/images \
  --labels data/labels.csv \
  --models densenet121-res224-all densenet121-res224-nih \
  --out-dir outputs/benchmark_v0_7 \
  --subgroups sex view_position \
  --save-plots \
  --selective
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from xraymind.config import DEFAULT_MODEL_NAME, MODEL_CHOICES
from xraymind.evaluation import (
    evaluate_predictions,
    evaluate_subgroups,
    label_columns,
    run_predictions,
    write_benchmark_model_card,
    write_dataset_card,
    write_run_manifest,
)
from xraymind.selective import (
    choose_operating_points,
    evaluate_selective_predictions,
    summarize_selective_curves,
    write_selective_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple XRayMind models on a labeled image folder")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", default="outputs/benchmark_v0_7")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL_NAME], choices=list(MODEL_CHOICES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-thresholds", action="store_true")
    parser.add_argument("--threshold-objective", choices=["f1", "youden"], default="f1")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--subgroups", nargs="*", default=[])
    parser.add_argument("--min-group-size", type=int, default=10)
    parser.add_argument("--dataset-name", default="XRayMind folder dataset")
    parser.add_argument("--selective", action="store_true", help="Also generate selective prediction / abstention artifacts")
    parser.add_argument("--max-risk", type=float, default=None, help="Optional aggregate selective-risk target")
    parser.add_argument("--min-coverage", type=float, default=None, help="Optional minimum automatic coverage target")
    return parser.parse_args()


def save_selective_curve_plot(summary_df: pd.DataFrame, out_path: Path) -> Path | None:
    if summary_df.empty:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(summary_df["coverage"], summary_df["mean_selective_risk"], marker="o", label="Mean selective risk")
    plt.xlabel("Coverage: fraction of cases automatically predicted")
    plt.ylabel("Selective risk: 1 - accuracy")
    plt.title("Selective prediction risk curve")
    plt.xlim(0, 1.05)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_df = pd.read_csv(args.labels)
    if args.limit:
        labels_df = labels_df.head(args.limit)

    labels = label_columns(labels_df, args.image_column, args.subgroups)
    all_metric_rows = []
    all_subgroup_rows = []
    all_selective_summary_rows = []
    prediction_paths = {}
    metric_paths = {}
    model_card_paths = {}
    selective_paths = {}

    dataset_card_path = write_dataset_card(
        labels_df,
        out_dir / "DATASET_CARD.md",
        image_column=args.image_column,
        subgroup_columns=args.subgroups,
        dataset_name=args.dataset_name,
    )

    for model_name in args.models:
        safe_model = model_name.replace("/", "_")
        model_dir = out_dir / safe_model
        model_dir.mkdir(parents=True, exist_ok=True)

        pred_df = run_predictions(
            image_dir=args.image_dir,
            labels_df=labels_df,
            image_column=args.image_column,
            model_name=model_name,
            limit=None,
            top_k=100,
            threshold=args.threshold,
        )
        pred_path = model_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        prediction_paths[model_name] = str(pred_path)

        plot_dir = model_dir / "reliability_plots" if args.save_plots else None
        metrics_df = evaluate_predictions(
            labels_df=labels_df,
            pred_df=pred_df,
            image_column=args.image_column,
            labels=labels,
            threshold=args.threshold,
            tune_thresholds=args.tune_thresholds,
            threshold_objective=args.threshold_objective,
            bootstrap=args.bootstrap,
            n_bins=args.n_bins,
            plot_dir=plot_dir,
        )
        metrics_df.insert(0, "model", model_name)
        metrics_path = model_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        metric_paths[model_name] = str(metrics_path)
        all_metric_rows.append(metrics_df)

        subgroup_path = None
        if args.subgroups:
            subgroup_df = evaluate_subgroups(
                labels_df=labels_df,
                pred_df=pred_df,
                subgroup_columns=args.subgroups,
                image_column=args.image_column,
                labels=labels,
                min_group_size=args.min_group_size,
                threshold=args.threshold,
                n_bins=args.n_bins,
            )
            if not subgroup_df.empty:
                subgroup_df.insert(0, "model", model_name)
                subgroup_path = model_dir / "subgroup_metrics.csv"
                subgroup_df.to_csv(subgroup_path, index=False)
                all_subgroup_rows.append(subgroup_df)

        if args.selective:
            selective_dir = model_dir / "selective_prediction"
            selective_dir.mkdir(parents=True, exist_ok=True)
            selective_curves = evaluate_selective_predictions(
                labels_df=labels_df,
                pred_df=pred_df,
                image_column=args.image_column,
                labels=labels,
                threshold=args.threshold,
            )
            selective_curves_path = selective_dir / "selective_curves.csv"
            selective_curves.to_csv(selective_curves_path, index=False)
            selective_summary = summarize_selective_curves(selective_curves)
            selective_summary.insert(0, "model", model_name)
            selective_summary_path = selective_dir / "selective_summary.csv"
            selective_summary.to_csv(selective_summary_path, index=False)
            all_selective_summary_rows.append(selective_summary)
            operating_point = choose_operating_points(
                selective_summary.drop(columns=["model"], errors="ignore"),
                max_risk=args.max_risk,
                min_coverage=args.min_coverage,
            )
            operating_path = selective_dir / "operating_point.json"
            operating_path.write_text(json.dumps(operating_point, indent=2), encoding="utf-8")
            curve_plot_path = save_selective_curve_plot(
                selective_summary.drop(columns=["model"], errors="ignore"),
                selective_dir / "selective_risk_curve.png",
            )
            selective_report_path = write_selective_report(
                output_path=selective_dir / "SELECTIVE_PREDICTION_REPORT.md",
                summary_df=selective_summary.drop(columns=["model"], errors="ignore"),
                operating_point=operating_point,
                dataset_name=args.dataset_name,
                model_name=model_name,
                plot_path=curve_plot_path.name if curve_plot_path else None,
            )
            selective_paths[model_name] = {
                "curves": str(selective_curves_path),
                "summary": str(selective_summary_path),
                "operating_point": str(operating_path),
                "plot": str(curve_plot_path) if curve_plot_path else None,
                "report": str(selective_report_path),
            }

        model_card_path = write_benchmark_model_card(
            metrics_df=metrics_df.drop(columns=["model"], errors="ignore"),
            output_path=model_dir / "MODEL_CARD.md",
            model_name=model_name,
            dataset_name=args.dataset_name,
            subgroup_metrics_path=subgroup_path,
        )
        model_card_paths[model_name] = str(model_card_path)

    combined_metrics = pd.concat(all_metric_rows, ignore_index=True) if all_metric_rows else pd.DataFrame()
    combined_metrics_path = out_dir / "combined_metrics.csv"
    combined_metrics.to_csv(combined_metrics_path, index=False)

    combined_subgroup_path = None
    if all_subgroup_rows:
        combined_subgroups = pd.concat(all_subgroup_rows, ignore_index=True)
        combined_subgroup_path = out_dir / "combined_subgroup_metrics.csv"
        combined_subgroups.to_csv(combined_subgroup_path, index=False)

    combined_selective_summary_path = None
    if all_selective_summary_rows:
        combined_selective_summary = pd.concat(all_selective_summary_rows, ignore_index=True)
        combined_selective_summary_path = out_dir / "combined_selective_summary.csv"
        combined_selective_summary.to_csv(combined_selective_summary_path, index=False)

    summary_path = out_dir / "leaderboard.csv"
    if not combined_metrics.empty:
        leaderboard = (
            combined_metrics.groupby("model")
            .agg(
                mean_auroc=("auroc", "mean"),
                mean_auprc=("auprc", "mean"),
                mean_brier=("brier", "mean"),
                mean_ece=("ece", "mean"),
                mean_f1=("f1", "mean"),
                labels_evaluated=("label", "count"),
            )
            .reset_index()
            .sort_values(["mean_auroc", "mean_auprc"], ascending=False)
        )
        leaderboard.to_csv(summary_path, index=False)

    manifest_path = write_run_manifest(
        out_dir / "run_manifest.json",
        dataset_name=args.dataset_name,
        image_dir=args.image_dir,
        labels_csv=args.labels,
        image_column=args.image_column,
        models=args.models,
        labels=labels,
        subgroup_columns=args.subgroups,
        threshold=args.threshold,
        tune_thresholds=args.tune_thresholds,
        threshold_objective=args.threshold_objective,
        bootstrap=args.bootstrap,
        selective=args.selective,
        max_risk=args.max_risk,
        min_coverage=args.min_coverage,
        outputs={
            "dataset_card": str(dataset_card_path),
            "combined_metrics": str(combined_metrics_path),
            "combined_subgroup_metrics": str(combined_subgroup_path) if combined_subgroup_path else None,
            "combined_selective_summary": str(combined_selective_summary_path) if combined_selective_summary_path else None,
            "leaderboard": str(summary_path),
            "predictions": prediction_paths,
            "metrics": metric_paths,
            "model_cards": model_card_paths,
            "selective_prediction": selective_paths,
        },
    )

    print(f"Saved dataset card to {dataset_card_path}")
    print(f"Saved combined metrics to {combined_metrics_path}")
    if combined_selective_summary_path:
        print(f"Saved combined selective summary to {combined_selective_summary_path}")
    print(f"Saved leaderboard to {summary_path}")
    print(f"Saved run manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
