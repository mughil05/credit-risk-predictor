import pandas as pd
import pytest

from src.data import load_and_preprocess
from src.model import (build_pipeline, evaluate_baseline, load_model,
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
    assert baseline == {
        "accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.5,
    }
