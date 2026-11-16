from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve,
                             brier_score_loss)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier

from src.schema import (FEATURES, TARGET, NUMERIC_FEATURES,
                        CATEGORICAL_FEATURES)

SEED = 42
THRESHOLD = 0.5
CALIBRATION_BINS = 10
MIN_SUBGROUP_ROWS = 20


def calibration_diagnostics(y_true, probability, n_bins=CALIBRATION_BINS):
    """Return Brier score and nonempty equal-frequency reliability bins."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if y_true.ndim != 1 or probability.ndim != 1 or y_true.shape != probability.shape:
        raise ValueError("y_true and probability must be same-length one-dimensional arrays")
    if len(y_true) == 0:
        raise ValueError("calibration requires at least one observation")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite and between 0 and 1")
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    # Quantile edges keep bins populated for continuous scores. Repeated edges
    # collapse naturally; a constant predictor becomes one fully-counted bin.
    edges = np.unique(np.quantile(probability, np.linspace(0, 1, n_bins + 1)))
    if len(edges) == 1:
        assignments = np.zeros(len(probability), dtype=int)
    else:
        assignments = np.searchsorted(edges[1:-1], probability, side="right")
    bins = []
    for index in np.unique(assignments):
        selected = assignments == index
        bins.append({
            "bin": int(index + 1),
            "mean_predicted_probability": float(probability[selected].mean()),
            "observed_bad_rate": float(y_true[selected].mean()),
            "count": int(selected.sum()),
        })
    return {
        "brier_score": float(brier_score_loss(y_true, probability)),
        "n_bins_requested": n_bins,
        "binning": "equal_frequency",
        "bins": bins,
    }


def calibration_display_data(metadata):
    """Read the exact saved calibration values rendered by the dashboard."""
    return {
        "logistic_regression": {
            "brier_score": metadata["brier_score"],
            "bins": metadata["reliability_bins"],
        },
        "majority_class_baseline": {
            "brier_score": metadata["baseline"]["brier_score"],
            "bins": metadata["baseline"]["reliability_bins"],
        },
        "training_prevalence_baseline": {
            "predicted_bad_probability": metadata["training_prevalence_baseline"]["predicted_bad_probability"],
            "brier_score": metadata["training_prevalence_baseline"]["brier_score"],
            "bins": metadata["training_prevalence_baseline"]["reliability_bins"],
        },
    }


def purpose_error_analysis(X_test, y_test, probability, min_rows=MIN_SUBGROUP_ROWS):
    """Descriptive held-out error summaries for sufficiently large purpose groups."""
    y_test = np.asarray(y_test, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(X_test) != len(y_test) or y_test.shape != probability.shape:
        raise ValueError("subgroup inputs must have the same number of rows")
    result = []
    purposes = X_test["purpose"].astype(str).to_numpy()
    predictions = (probability >= THRESHOLD).astype(int)
    for purpose in sorted(set(purposes)):
        selected = purposes == purpose
        count = int(selected.sum())
        if count < min_rows:
            continue
        actual, predicted = y_test[selected], probability[selected]
        labels = predictions[selected]
        result.append({
            "purpose": purpose,
            "count": count,
            "bad_count": int(actual.sum()),
            "observed_bad_rate": float(actual.mean()),
            "mean_predicted_bad_probability": float(predicted.mean()),
            "brier_score": float(brier_score_loss(actual, predicted)),
            "false_positives_at_0_50": int(((labels == 1) & (actual == 0)).sum()),
            "false_negatives_at_0_50": int(((labels == 0) & (actual == 1)).sum()),
        })
    return result


def threshold_metrics(y_true, probability, threshold=THRESHOLD):
    """Calculate binary classification metrics from saved bad-risk probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if y_true.shape != probability.shape:
        raise ValueError("y_true and probability must have the same shape")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def evaluate_baseline(X_train, y_train, X_test, y_test):
    """Evaluate a majority-class baseline; zero_division makes no-positive scores 0."""
    y_train, y_test = np.asarray(y_train, dtype=int), np.asarray(y_test, dtype=int)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    prediction = dummy.predict(X_test)
    bad_column = int(np.flatnonzero(dummy.classes_ == 1)[0])
    probability = dummy.predict_proba(X_test)[:, bad_column]
    calibration = calibration_diagnostics(y_test, probability)
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        # A constant score has no ranking information; ROC-AUC is 0.5 due to ties.
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "brier_score": calibration["brier_score"],
        "test_probabilities": probability.astype(float).tolist(),
        "reliability_bins": calibration["bins"],
    }


def evaluate_training_prevalence_baseline(y_train, y_test):
    """Score the training bad-credit prevalence as a constant-probability reference."""
    y_train, y_test = np.asarray(y_train, dtype=int), np.asarray(y_test, dtype=int)
    if y_train.ndim != 1 or y_test.ndim != 1 or len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("training and holdout targets must be nonempty one-dimensional arrays")
    if not np.isin(y_train, [0, 1]).all() or not np.isin(y_test, [0, 1]).all():
        raise ValueError("targets must contain only 0 and 1")
    prevalence = float(y_train.mean())
    probability = np.full(len(y_test), prevalence, dtype=float)
    calibration = calibration_diagnostics(y_test, probability)
    return {
        "kind": "constant_probability_reference",
        "predicted_bad_probability": prevalence,
        "brier_score": calibration["brier_score"],
        "reliability_bins": calibration["bins"],
    }


def build_pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES),
                                    ("categorical", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocess", preprocess),
                     ("classifier", LogisticRegression(max_iter=2000, random_state=SEED))])


def train(data, model_path="artifacts/model.joblib", metadata_path="artifacts/metadata.json"):
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    fixed_metrics = threshold_metrics(y_test, probability, THRESHOLD)
    calibration = calibration_diagnostics(y_test, probability)
    fpr, tpr, _ = roc_curve(y_test, probability)
    baseline = evaluate_baseline(X_train, y_train, X_test, y_test)
    prevalence_baseline = evaluate_training_prevalence_baseline(y_train, y_test)
    metrics = {
        **fixed_metrics,
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "class_balance": {"good (0)": int((y == 0).sum()), "bad (1)": int((y == 1).sum())},
        "test_class_balance": {"good (0)": int((y_test == 0).sum()), "bad (1)": int((y_test == 1).sum())},
        "threshold": THRESHOLD,
        "seed": SEED,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "test_targets": y_test.astype(int).tolist(),
        "test_probabilities": probability.astype(float).tolist(),
        "brier_score": calibration["brier_score"],
        "reliability_bins": calibration["bins"],
        "calibration_binning": calibration["binning"],
        "purpose_error_analysis": purpose_error_analysis(X_test, y_test, probability),
        "baseline": baseline,
        "training_prevalence_baseline": prevalence_baseline,
    }
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return pipeline, metrics


def load_model(path="artifacts/model.joblib"):
    return joblib.load(path)


def coefficient_frame(pipeline):
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    values = pipeline.named_steps["classifier"].coef_[0]
    import pandas as pd
    return pd.DataFrame({"feature": names, "coefficient": values}).sort_values("coefficient")

