import pandas as pd

from xraymind.ensemble import ensemble_from_prediction_frames, summarize_ensemble_uncertainty
from xraymind.selective import evaluate_selective_predictions, summarize_selective_curves


def test_ensemble_from_prediction_frames_adds_mean_and_uncertainty_columns():
    model_a = pd.DataFrame(
        {
            "image": ["a.png", "b.png"],
            "Cardiomegaly": [0.8, 0.4],
            "Effusion": [0.2, 0.7],
        }
    )
    model_b = pd.DataFrame(
        {
            "image": ["a.png", "b.png"],
            "Cardiomegaly": [0.6, 0.5],
            "Effusion": [0.4, 0.9],
        }
    )

    ensemble = ensemble_from_prediction_frames(
        {"model_a": model_a, "model_b": model_b},
        labels=["Cardiomegaly", "Effusion"],
    )

    assert list(ensemble["image"]) == ["a.png", "b.png"]
    assert ensemble.loc[0, "Cardiomegaly"] == 0.7
    assert "Cardiomegaly_ensemble_std" in ensemble.columns
    assert "Cardiomegaly_ensemble_uncertainty" in ensemble.columns
    assert "Cardiomegaly_confidence" in ensemble.columns
    assert "Cardiomegaly__model_a" in ensemble.columns
    assert "Cardiomegaly__model_b" in ensemble.columns


def test_summarize_ensemble_uncertainty_returns_per_label_rows():
    ensemble = pd.DataFrame(
        {
            "image": ["a.png", "b.png"],
            "Cardiomegaly": [0.7, 0.45],
            "Cardiomegaly_ensemble_uncertainty": [0.1, 0.05],
            "Cardiomegaly_ensemble_std": [0.1, 0.05],
            "Cardiomegaly_ensemble_range": [0.2, 0.1],
            "Cardiomegaly_ensemble_entropy": [0.61, 0.69],
            "Cardiomegaly_confidence": [0.8, 0.6],
        }
    )

    summary = summarize_ensemble_uncertainty(ensemble, labels=["Cardiomegaly"])

    assert len(summary) == 1
    assert summary.loc[0, "label"] == "Cardiomegaly"
    assert summary.loc[0, "n"] == 2
    assert summary.loc[0, "mean_uncertainty"] == 0.07500000000000001


def test_selective_prediction_can_use_ensemble_confidence_columns():
    labels = pd.DataFrame(
        {
            "image": ["a.png", "b.png", "c.png", "d.png"],
            "Cardiomegaly": [1, 0, 1, 0],
        }
    )
    predictions = pd.DataFrame(
        {
            "image": ["a.png", "b.png", "c.png", "d.png"],
            "Cardiomegaly": [0.9, 0.1, 0.55, 0.45],
            "Cardiomegaly_confidence": [0.95, 0.9, 0.4, 0.3],
            "Cardiomegaly_ensemble_uncertainty": [0.01, 0.02, 0.4, 0.5],
        }
    )

    curves = evaluate_selective_predictions(
        labels,
        predictions,
        labels=["Cardiomegaly"],
        coverage_grid=[0.5, 1.0],
        uncertainty_suffix="_ensemble_uncertainty",
    )
    summary = summarize_selective_curves(curves)

    assert not curves.empty
    assert set(summary["coverage"]) == {0.5, 1.0}
    assert curves.loc[curves["coverage"] == 0.5, "selective_accuracy"].iloc[0] == 1.0
