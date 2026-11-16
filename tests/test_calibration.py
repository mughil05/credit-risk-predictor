import numpy as np
import pandas as pd
import pytest

from src.model import (calibration_display_data, calibration_diagnostics,
                       evaluate_baseline, evaluate_training_prevalence_baseline,
                       purpose_error_analysis)


def test_brier_score_and_equal_frequency_bin_counts():
    actual = [0, 1, 1, 0]
    probability = [0.1, 0.4, 0.8, 0.9]
    result = calibration_diagnostics(actual, probability, n_bins=2)
    assert result["brier_score"] == pytest.approx(0.305)
    assert result["binning"] == "equal_frequency"
    assert [row["count"] for row in result["bins"]] == [2, 2]
    assert sum(row["count"] for row in result["bins"]) == len(actual)
    assert [row["mean_predicted_probability"] for row in result["bins"]] == pytest.approx([0.25, 0.85])
    assert [row["observed_bad_rate"] for row in result["bins"]] == pytest.approx([0.5, 0.5])
    assert all(0 <= row["mean_predicted_probability"] <= 1 for row in result["bins"])
    assert all(0 <= row["observed_bad_rate"] <= 1 for row in result["bins"])


def test_calibration_rejects_out_of_bounds_and_nonfinite_probabilities():
    actual = [0, 1, 1, 0]
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibration_diagnostics(actual, [0.1, 0.4, 1.01, 0.9])
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibration_diagnostics(actual, [0.1, 0.4, np.nan, 0.9])


def test_constant_majority_baseline_uses_actual_predicted_bad_probabilities():
    baseline = evaluate_baseline([[1], [2], [3], [4]], [0, 0, 0, 1],
                                 [[5], [6], [7], [8]], [0, 1, 0, 1])
    assert baseline["test_probabilities"] == [0.0, 0.0, 0.0, 0.0]
    assert baseline["brier_score"] == pytest.approx(0.5)
    assert baseline["reliability_bins"] == [{
        "bin": 1,
        "mean_predicted_probability": 0.0,
        "observed_bad_rate": 0.5,
        "count": 4,
    }]


def test_training_prevalence_reference_uses_training_targets_only():
    reference = evaluate_training_prevalence_baseline(
        y_train=[0, 0, 0, 1],
        y_test=[0, 0, 0, 0],
    )
    assert reference["kind"] == "constant_probability_reference"
    assert reference["predicted_bad_probability"] == pytest.approx(0.25)
    assert reference["brier_score"] == pytest.approx(0.0625)
    assert reference["reliability_bins"] == [{
        "bin": 1,
        "mean_predicted_probability": 0.25,
        "observed_bad_rate": 0.0,
        "count": 4,
    }]


def test_dashboard_calibration_values_come_from_saved_evaluation_data():
    saved = {
        "brier_score": 0.22,
        "reliability_bins": [{"mean_predicted_probability": 0.2,
                              "observed_bad_rate": 0.25, "count": 4}],
        "baseline": {"brier_score": 0.3,
                     "reliability_bins": [{"mean_predicted_probability": 0.0,
                                           "observed_bad_rate": 0.3, "count": 10}]},
        "training_prevalence_baseline": {
            "predicted_bad_probability": 0.3,
            "brier_score": 0.21,
            "reliability_bins": [{"mean_predicted_probability": 0.3,
                                  "observed_bad_rate": 0.3, "count": 10}],
        },
    }
    display = calibration_display_data(saved)
    assert display["logistic_regression"]["brier_score"] == saved["brier_score"]
    assert display["logistic_regression"]["bins"] == saved["reliability_bins"]
    assert display["majority_class_baseline"]["brier_score"] == saved["baseline"]["brier_score"]
    assert display["majority_class_baseline"]["bins"] == saved["baseline"]["reliability_bins"]
    assert display["training_prevalence_baseline"]["brier_score"] == saved["training_prevalence_baseline"]["brier_score"]
    assert display["training_prevalence_baseline"]["predicted_bad_probability"] == saved["training_prevalence_baseline"]["predicted_bad_probability"]
    assert display["training_prevalence_baseline"]["bins"] == saved["training_prevalence_baseline"]["reliability_bins"]


def test_purpose_error_analysis_omits_sparse_categories_and_reports_counts():
    frame = pd.DataFrame({"purpose": ["A"] * 20 + ["B"] * 4})
    actual = np.array([0] * 10 + [1] * 10 + [0, 1, 0, 1])
    probability = np.array([0.2] * 10 + [0.8] * 10 + [0.3] * 4)
    result = purpose_error_analysis(frame, actual, probability, min_rows=20)
    assert len(result) == 1
    assert result[0]["purpose"] == "A"
    assert result[0]["count"] == 20
    assert result[0]["bad_count"] == 10
    assert result[0]["false_positives_at_0_50"] == 0
    assert result[0]["false_negatives_at_0_50"] == 0
