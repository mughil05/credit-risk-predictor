from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier

from src.schema import (FEATURES, TARGET, NUMERIC_FEATURES,
                        CATEGORICAL_FEATURES)

SEED = 42
THRESHOLD = 0.5


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
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        # A constant score has no ranking information; ROC-AUC is 0.5 due to ties.
        "roc_auc": float(roc_auc_score(y_test, probability)),
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
    fpr, tpr, _ = roc_curve(y_test, probability)
    baseline = evaluate_baseline(X_train, y_train, X_test, y_test)
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
        "baseline": baseline,
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

