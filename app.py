import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.model import (calibration_display_data, coefficient_frame, load_model,
                       threshold_metrics)
from src.schema import FEATURES, RAW_FEATURES, NUMERIC_FEATURES

CATEGORIES = {
    "checking_account_status": ["A11", "A12", "A13", "A14"],
    "credit_history": ["A30", "A31", "A32", "A33", "A34"],
    "purpose": ["A40", "A41", "A410", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49"],
    "savings_account_bonds": ["A61", "A62", "A63", "A64", "A65"],
    "employment_since": ["A71", "A72", "A73", "A74", "A75"],
    "personal_status_sex": ["A91", "A92", "A93", "A94", "A95"],
    "other_debtors_guarantors": ["A101", "A102", "A103"],
    "property": ["A121", "A122", "A123", "A124"],
    "other_installment_plans": ["A141", "A142", "A143"],
    "housing": ["A151", "A152", "A153"],
    "job": ["A171", "A172", "A173", "A174"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"],
}

st.set_page_config(page_title="Credit Risk Predictor", page_icon="📊", layout="wide")
st.title("German Credit Risk Predictor")
st.caption("Educational model estimates based on the UCI Statlog German Credit dataset. Predictions are not lending decisions.")

model_file, metadata_file = Path("artifacts/model.joblib"), Path("artifacts/metadata.json")
if not model_file.exists() or not metadata_file.exists():
    st.error("Model artifacts are missing. Run `python train.py` first.")
    st.stop()
model = load_model(model_file)
metadata = json.loads(metadata_file.read_text())

st.subheader("Model overview")
st.write(
    "This project uses 1,000 historical German credit applications from UCI. "
    "The target is good (0) versus bad credit risk (1), with bad risk treated as positive. "
    "A logistic regression pipeline is evaluated on a stratified, held-out 25% test set; "
    "imputation, encoding, and scaling are fitted only on training rows."
)
st.caption(f"Dataset balance: {metadata['class_balance']['good (0)']} good / {metadata['class_balance']['bad (1)']} bad. Test set: {metadata['test_rows']} rows.")

st.subheader("Original held-out evaluation · threshold 0.50")
st.caption("These fixed results are the reported evaluation. The threshold controls below recompute classifications from the same saved test probabilities; ROC-AUC does not change.")
performance = pd.DataFrame([
    {"Model": "Logistic regression", **{key: metadata[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc")}},
    {"Model": "Majority-class baseline", **{key: metadata["baseline"][key]
                                               for key in ("accuracy", "precision", "recall", "f1", "roc_auc")}},
]).set_index("Model")
st.dataframe(performance.style.format("{:.3f}"), use_container_width=True)
st.caption("Accuracy = share classified correctly. Precision = share of predicted bad-risk cases that are actually bad. Recall = share of actual bad-risk cases identified. ROC-AUC = ability to rank bad-risk cases above good-risk cases across thresholds.")

st.subheader("Held-out probability calibration")
calibration = calibration_display_data(metadata)
cal_cols = st.columns(3)
cal_cols[0].metric("Logistic regression · Brier score",
                   f"{calibration['logistic_regression']['brier_score']:.3f}")
cal_cols[1].metric("Majority baseline · Brier score",
                   f"{calibration['majority_class_baseline']['brier_score']:.3f}")
cal_cols[2].metric("Training-prevalence reference · Brier score",
                   f"{calibration['training_prevalence_baseline']['brier_score']:.3f}")
baseline_bin = calibration["majority_class_baseline"]["bins"][0]
st.caption(
    "The majority baseline's saved probability for bad credit is "
    f"{baseline_bin['mean_predicted_probability']:.3f}; observed bad frequency is "
    f"{baseline_bin['observed_bad_rate']:.3f} across {baseline_bin['count']} test cases."
)
prevalence = calibration["training_prevalence_baseline"]
st.caption(
    "The training-prevalence reference assigns the training-set bad-credit proportion "
    f"({prevalence['predicted_bad_probability']:.3f}) to every holdout case. "
    "It is a constant-probability reference, not a fitted model."
)
st.caption(
    "Discrimination (ROC-AUC) measures how well scores rank bad-risk cases above good-risk cases. "
    "Calibration measures whether a predicted probability matches the observed frequency; "
    "lower Brier scores indicate better probability accuracy. Neither metric establishes fairness or future performance."
)
reliability = pd.DataFrame(calibration["logistic_regression"]["bins"])
reliability = reliability.set_index("mean_predicted_probability")
reliability = reliability.rename(columns={"observed_bad_rate": "Observed bad frequency"})
reliability["Perfect calibration"] = reliability.index.astype(float)
st.markdown("**Reliability diagram · logistic regression**")
chart_reliability = reliability[["Observed bad frequency", "Perfect calibration"]].copy()
chart_reliability.index.name = "Mean predicted bad-credit probability"
st.line_chart(chart_reliability)
st.caption("Quantile bins target roughly equal sample counts. Counts are shown because bin frequencies can still be noisy.")
st.dataframe(reliability.reset_index()[[
    "mean_predicted_probability", "Observed bad frequency", "count"
]].rename(columns={"mean_predicted_probability": "Mean predicted probability",
                    "Observed bad frequency": "Observed bad frequency",
                    "count": "Applicants in bin"}),
    hide_index=True, use_container_width=True)

subgroups = metadata.get("purpose_error_analysis", [])
if subgroups:
    with st.expander("Exploratory held-out errors by credit purpose"):
        st.caption(
            "Descriptive model-error summaries for purpose categories with at least 20 test cases. "
            "These small groups are not a fairness analysis and do not support conclusions about applicants or lending impact."
        )
        st.dataframe(pd.DataFrame(subgroups), hide_index=True, use_container_width=True)
else:
    st.caption("No credit-purpose subgroup met the 20-case minimum on this holdout; subgroup error summaries are omitted.")

with st.expander("Original held-out confusion matrix at 0.50"):
    st.dataframe(pd.DataFrame(metadata["confusion_matrix"], index=["Actual good", "Actual bad"],
                              columns=["Predicted good", "Predicted bad"]), use_container_width=True)

st.subheader("Explore the classification threshold")
threshold = st.slider("Bad-risk probability threshold", min_value=0.10, max_value=0.90,
                      value=float(metadata["threshold"]), step=0.01,
                      help="Lower thresholds identify more bad-risk cases, usually with more good-risk applicants flagged. This changes classifications only; it does not retrain the model.")
selected = threshold_metrics(metadata["test_targets"], metadata["test_probabilities"], threshold)
st.caption(f"Adjustable metrics at threshold {threshold:.2f}, calculated from the saved held-out predictions. The 0.50 report above remains the original evaluation.")
selected_cols = st.columns(4)
for col, label, key in zip(selected_cols, ["Accuracy", "Precision · bad risk", "Recall · bad risk", "F1 · bad risk"],
                           ["accuracy", "precision", "recall", "f1"]):
    col.metric(label, f"{selected[key]:.3f}")
left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix at selected threshold**  \nRows = actual class; columns = predicted class.")
    st.dataframe(pd.DataFrame(selected["confusion_matrix"], index=["Actual good", "Actual bad"],
                              columns=["Predicted good", "Predicted bad"]), use_container_width=True)
with right:
    thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)
    curve_rows = [threshold_metrics(metadata["test_targets"], metadata["test_probabilities"], t)
                  for t in thresholds]
    pr_curve = pd.DataFrame({"Precision (bad risk)": [row["precision"] for row in curve_rows],
                             "Recall (bad risk)": [row["recall"] for row in curve_rows]},
                            index=pd.Index(thresholds, name="Probability threshold"))
    st.markdown("**Precision and recall by threshold**")
    st.line_chart(pr_curve)
st.info("A lower threshold tends to catch more bad-risk borrowers (higher recall), while also flagging more good-risk borrowers (lower precision). A higher threshold usually reduces false alarms but misses more bad-risk cases. Choose thresholds based on the intended use and costs, not by optimizing this test set.")

st.markdown(f"**ROC curve** · ROC-AUC {metadata['roc_auc']:.3f}")
st.caption("ROC-AUC summarizes ranking across all thresholds. It remains unchanged when you move the threshold slider.")
roc_points = pd.DataFrame({"False positive rate": metadata["roc_curve"]["fpr"],
                           "True positive rate": metadata["roc_curve"]["tpr"]}).set_index("False positive rate")
roc_points.index.name = "False positive rate"
st.line_chart(roc_points)

st.subheader("Logistic regression coefficients")
st.caption("Positive coefficients increase the model's estimated log-odds of bad risk, conditional on the encoded features. They are not causal effects.")
coefs = coefficient_frame(model)
st.bar_chart(coefs.set_index("feature")["coefficient"])

st.subheader("Estimate a borrower profile")
with st.form("borrower"):
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(RAW_FEATURES):
        with cols[i % 2]:
            if feature in NUMERIC_FEATURES:
                defaults = {"duration_months": (6, 72, 24), "credit_amount": (250, 20000, 3000),
                            "installment_rate_percent": (1, 4, 2), "residence_since_years": (1, 4, 2),
                            "age_years": (18, 100, 35), "existing_credits_count": (1, 4, 1),
                            "dependents_count": (1, 2, 1)}
                low, high, default = defaults[feature]
                values[feature] = st.number_input(feature.replace("_", " ").title(), low, high, default)
            else:
                options = CATEGORIES[feature]
                values[feature] = st.selectbox(feature.replace("_", " ").title(), options)
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    values["credit_amount_per_month"] = values["credit_amount"] / values["duration_months"]
    probability = float(model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1])
    label = "Bad risk estimate" if probability >= threshold else "Good risk estimate"
    st.metric("Estimated probability of bad credit risk", f"{probability:.1%}")
    st.info(f"Classification at selected threshold {threshold:.2f}: **{label}**. This model output is an estimate, not an approval, denial, or lending decision.")
