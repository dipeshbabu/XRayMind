import numpy as np
import pandas as pd

from xraymind.conformal import (
    apply_conformal_sets,
    binary_nonconformity_scores,
    calibrate_conformal_thresholds,
    conformal_quantile,
    evaluate_conformal_sets,
    split_calibration_eval,
)


def test_binary_nonconformity_scores_positive_and_negative():
    scores = binary_nonconformity_scores([1, 0, 1], [0.9, 0.2, 0.4])
    assert np.allclose(scores, [0.1, 0.2, 0.6])


def test_conformal_quantile_is_conservative_order_statistic():
    qhat = conformal_quantile([0.1, 0.2, 0.3, 0.4], alpha=0.25)
    assert qhat == 0.4


def test_calibrate_apply_and_evaluate_conformal_sets():
    labels_df = pd.DataFrame(
        {
            "image": ["a.png", "b.png", "c.png", "d.png"],
            "Finding": [1, 0, 1, 0],
        }
    )
    pred_df = pd.DataFrame(
        {
            "image": ["a.png", "b.png", "c.png", "d.png"],
            "Finding": [0.95, 0.10, 0.70, 0.20],
        }
    )
    thresholds = calibrate_conformal_thresholds(labels_df, pred_df, labels=["Finding"], alpha=0.1)
    assert list(thresholds["label"]) == ["Finding"]
    assert "qhat" in thresholds.columns

    conformal = apply_conformal_sets(pred_df, thresholds)
    assert "Finding_conformal_set" in conformal.columns
    assert "Finding_conformal_set_size" in conformal.columns

    summary = evaluate_conformal_sets(labels_df, conformal, thresholds)
    assert summary.loc[0, "label"] == "Finding"
    assert 0.0 <= summary.loc[0, "empirical_coverage"] <= 1.0
    assert summary.loc[0, "mean_set_size"] >= 0.0


def test_split_calibration_eval_is_reproducible_and_nonempty():
    df = pd.DataFrame({"image": [f"{i}.png" for i in range(10)], "Finding": [0, 1] * 5})
    cal1, eval1 = split_calibration_eval(df, calibration_fraction=0.4, seed=7)
    cal2, eval2 = split_calibration_eval(df, calibration_fraction=0.4, seed=7)
    assert len(cal1) == 4
    assert len(eval1) == 6
    assert cal1["image"].tolist() == cal2["image"].tolist()
    assert eval1["image"].tolist() == eval2["image"].tolist()
