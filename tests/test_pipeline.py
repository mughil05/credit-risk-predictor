import json

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data import load_and_preprocess
from src.model import (build_pipeline, calibration_display_data,
                       calibration_diagnostics, evaluate_baseline, load_model,
                       threshold_metrics, train)
from src.schema import FEATURES, TARGET, TARGET_MAPPING


@pytest.fixture(scope="module")
def data():
    return load_and_preprocess("data/german.data")


def test_target_mapping_and_original_row_count(data):
    assert TARGET_MAPPING == {1: 0, 2: 1}
    assert len(data) == 1000
    assert set(data[TARGET].unique()) == {0, 1}
    assert data[TARGET].value_counts().to_dict() == {0: 700, 1: 300}


def test_spark_preprocessing_and_feature_types(data):
    assert list(data.columns) == FEATURES + [TARGET]
    assert data["duration_months"].dtype.kind in "fi"
    assert data["purpose"].dtype == object
    assert not data.isna().any().any()


def test_feature_transformations_fit_and_handle_unknown_categories(data):
    pipeline = build_pipeline()
    pipeline.fit(data[FEATURES], data[TARGET])
    transformed = pipeline.named_steps["preprocess"].transform(data[FEATURES].iloc[:3])
    assert transformed.shape[0] == 3
    unknown = data[FEATURES].iloc[[0]].copy()
    unknown["purpose"] = "UNSEEN"
    assert pipeline.predict_proba(unknown).shape == (1, 2)


def test_training_saves_loadable_model_and_inference(data, tmp_path):
    model_path, meta_path = tmp_path / "model.joblib", tmp_path / "metadata.json"
    model, metrics = train(data, model_path, meta_path)
    reloaded = load_model(model_path)
    p = reloaded.predict_proba(data[FEATURES].iloc[:4])
    assert p.shape == (4, 2)
    assert (p >= 0).all() and (p <= 1).all()
    assert (abs(p.sum(axis=1) - 1) < 1e-8).all()
    assert metrics["confusion_matrix"] and meta_path.exists()
    assert len(metrics["test_targets"]) == len(metrics["test_probabilities"]) == 250
    assert metrics["baseline"]["accuracy"] == 0.7
    assert metrics["brier_score"] == pytest.approx(
        calibration_diagnostics(metrics["test_targets"], metrics["test_probabilities"])["brier_score"]
    )
    assert metrics["baseline"]["brier_score"] == pytest.approx(
        calibration_diagnostics(metrics["test_targets"], metrics["baseline"]["test_probabilities"])["brier_score"]
    )
    prevalence = metrics["training_prevalence_baseline"]
    _, _, y_train, _ = train_test_split(data[FEATURES], data[TARGET],
                                        test_size=0.25, random_state=metrics["seed"],
                                        stratify=data[TARGET])
    assert prevalence["predicted_bad_probability"] == pytest.approx(float(y_train.mean()))
    assert prevalence["brier_score"] == pytest.approx(
        calibration_diagnostics(metrics["test_targets"],
                                [prevalence["predicted_bad_probability"]] * metrics["test_rows"])["brier_score"]
    )
    saved_metrics = json.loads(meta_path.read_text())
    assert calibration_display_data(saved_metrics) == calibration_display_data(metrics)
    assert sum(row["count"] for row in metrics["reliability_bins"]) == 250
    assert all(row["count"] >= 20 for row in metrics["purpose_error_analysis"])
    assert len(metrics["baseline"]["test_probabilities"]) == 250


def test_threshold_metrics_recompute_from_probabilities():
    actual = [0, 0, 1, 1]
    probabilities = [0.2, 0.6, 0.4, 0.8]
    at_half = threshold_metrics(actual, probabilities, 0.5)
    at_point_three = threshold_metrics(actual, probabilities, 0.3)
    assert at_half["confusion_matrix"] == [[1, 1], [1, 1]]
    assert at_half["precision"] == 0.5
    assert at_half["recall"] == 0.5
    assert at_half["f1"] == 0.5
    assert at_point_three["confusion_matrix"] == [[1, 1], [0, 2]]
    assert at_point_three["recall"] == 1.0
    assert at_point_three["precision"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        threshold_metrics(actual, probabilities, 1.1)


def test_majority_baseline_uses_training_majority_and_defined_metrics():
    baseline = evaluate_baseline([[1], [2], [3], [4]], [0, 0, 0, 1],
                                 [[5], [6], [7], [8]], [0, 1, 0, 1])
    assert {key: baseline[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc")} == {
        "accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.5,
    }
    assert baseline["test_probabilities"] == [0.0, 0.0, 0.0, 0.0]
    assert baseline["brier_score"] == pytest.approx(0.5)
