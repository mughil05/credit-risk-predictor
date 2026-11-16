# German Credit Risk Predictor

A small local machine-learning project that estimates whether an applicant is in the **bad credit risk** class using logistic regression. It uses the original 1,000-row UCI Statlog German Credit data, Spark for loading and basic cleaning, scikit-learn for leakage-safe modeling, and Streamlit for exploration and individual estimates.

**Tech stack:** Python, PySpark, pandas, scikit-learn, Streamlit.


## At a glance

- **Task:** Binary classification of bad credit risk using 1,000 historical UCI applications.
- **Held-out performance:** 0.805 ROC-AUC and 77.6% accuracy on 250 examples (versus 0.500 ROC-AUC and 70.0% accuracy for a majority-class baseline).
- **What to explore:** Model comparison, ROC curve, logistic regression coefficients, and a threshold slider that updates precision, recall, F1, and the confusion matrix.
- **Try a prediction:** Enter borrower attributes in the Streamlit form to see a model-estimated bad-risk probability. This is an educational demonstration, not a lending decision.


## Dataset and labels

Source: [UCI Machine Learning Repository — Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), donated by Hans Hofmann. Citation: Hofmann, H. (1994), *Statlog (German Credit Data)*, UCI Machine Learning Repository, [doi:10.24432/C5NC77](https://doi.org/10.24432/C5NC77). UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which permits redistribution with attribution. The original whitespace-delimited `german.data` is checked into `data/` so training is reproducible offline. It contains 20 attributes followed by a class label. No observations have been synthesized or manually changed. The source's `german.doc` describes the attributes and codes; this README documents each ordered feature and meaning.

The checked-in source file's SHA-256 is `b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871`.

The original target is `1 = good` and `2 = bad`. The project explicitly maps it to `good = 0`, `bad = 1`; therefore bad risk is the positive class in all metrics and predictions.

| Position | Feature | Meaning |
|---:|---|---|
| 1 | `checking_account_status` | Existing checking account status |
| 2 | `duration_months` | Credit duration in months |
| 3 | `credit_history` | Credit history |
| 4 | `purpose` | Credit purpose |
| 5 | `credit_amount` | Credit amount |
| 6 | `savings_account_bonds` | Savings account/bonds |
| 7 | `employment_since` | Present employment since |
| 8 | `installment_rate_percent` | Installment rate as percent of disposable income |
| 9 | `personal_status_sex` | Personal status and sex (source category) |
| 10 | `other_debtors_guarantors` | Other debtors or guarantors |
| 11 | `residence_since_years` | Present residence since |
| 12 | `property` | Property |
| 13 | `age_years` | Age in years |
| 14 | `other_installment_plans` | Other installment plans |
| 15 | `housing` | Housing |
| 16 | `existing_credits_count` | Number of existing credits at this bank |
| 17 | `job` | Job category |
| 18 | `dependents_count` | Number of people liable for maintenance |
| 19 | `telephone` | Telephone availability/category |
| 20 | `foreign_worker` | Foreign worker indicator/category |

The pipeline also derives `credit_amount_per_month = credit_amount / duration_months`. This is a deterministic feature calculation from original fields, not a changed observation.

Categorical values are the original UCI codes (`Axx`); numeric quantities are parsed as numbers. The original dataset has 700 good and 300 bad cases (70%/30%).

## Architecture

```text
app.py                 Streamlit dashboard
train.py               preprocessing + training entry point
src/data.py            Spark loading and preprocessing
src/model.py           split, pipeline, evaluation, serialization
src/schema.py          source and engineered feature definitions
data/german.data       original UCI data (1,000 rows)
tests/test_pipeline.py automated checks
requirements.txt       pinned direct dependencies
README.md              setup, methodology, results, limitations
```

- `src/data.py`: local SparkSession, whitespace parsing, type conversion, empty-to-null cleaning, amount-per-month feature derivation, target mapping, then Spark shutdown.
- `src/model.py`: stratified 75/25 split (seed 42), train-only median imputation and standard scaling for numeric inputs, train-only most-frequent imputation and one-hot encoding for categorical inputs, then logistic regression. All transformations live in a scikit-learn `Pipeline`/`ColumnTransformer`.
- `train.py`: produces `artifacts/model.joblib` and `artifacts/metadata.json`, including held-out targets/probabilities, metrics, ROC curve points, calibration bins/Brier scores, descriptive purpose-group errors, and majority-class baseline results.
- `app.py`: Streamlit overview, held-out logistic-vs-baseline comparison, calibration diagnostics, threshold slider, precision/recall chart, confusion matrices, ROC curve, coefficient chart, and borrower estimate form.
- `tests/`: data, target, transform, threshold metrics, calibration/Brier/bin counts, baseline evaluation, probability, serialization, and inference checks.

The reported classification threshold is **0.50** on the estimated probability of bad risk. The dashboard slider explores 0.10–0.90 by applying a threshold to saved held-out probabilities; it does not retrain or optimize a threshold on test outcomes. Precision, recall, and the confusion matrices treat bad risk as positive. No preprocessing parameters are fitted using test data. The `DummyClassifier(strategy="most_frequent")` baseline learns its majority label on the training split only and uses the identical test split.

## Setup and run

Requires Python 3.10–3.12 and Java 8 or newer for the included PySpark version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python -m pytest
streamlit run app.py
```

The source dataset is already included. To refresh it from UCI, download the original `german.data` from the source link above and replace `data/german.data` without editing its records. Do not publish generated model artifacts as ground truth; the app reads the local artifacts generated by training.

## Actual held-out evaluation and baseline

Generated from the current dataset by `python train.py`; the stratified test split has 250 rows (175 good, 75 bad). Results are recorded from the saved evaluation metadata and are reproducible with the fixed seed and dependency set. At threshold 0.50:

| Metric | Logistic regression | Majority-class baseline |
|---|---:|---:|
| Accuracy | 0.776 | 0.700 |
| Precision (bad positive) | 0.651 | 0.000 |
| Recall (bad positive) | 0.547 | 0.000 |
| F1 (bad positive) | 0.594 | 0.000 |
| ROC-AUC | 0.805 | 0.500 |

Confusion matrix (actual rows and predicted columns; good, bad): `[[153, 22], [34, 41]]`.

Undefined precision, recall, and F1 for a baseline with no predicted bad cases are reported as 0. ROC-AUC for its constant score is 0.5, reflecting no ranking ability.

### Probability calibration diagnostics

The held-out logistic regression Brier score is **0.155**. The unchanged
majority-class DummyClassifier Brier score is **0.300**. A separate
training-prevalence reference has Brier score **0.210**: it predicts the
training-set bad-credit proportion (0.300) for every holdout row. This is a
constant-probability reference, not a fitted machine-learning model. Its
probability comes from training labels only. Lower Brier scores mean lower
average squared error between predicted bad-credit probabilities and observed
0/1 outcomes.

The DummyClassifier comparison remains unchanged: its actual
`predict_proba` output for bad credit is 0.0 for every test case because it
predicts the training majority class, good credit. It is not assigned a 30%
prevalence probability.

The reliability diagram uses 10 equal-frequency bins from the same 250-row
holdout, with 25 applicants per bin. It compares each bin's mean predicted bad
probability with the observed bad-credit rate; the dashboard also displays each
bin's sample count. Across the bins, observed frequencies broadly move with
predicted probabilities, while individual bins fluctuate: for example, bins
near 0.05 and 0.09 have observed bad rates of 0.12 and 0.16, while a bin near
0.13 has an observed rate of 0.04. With only 25 cases per bin, these differences
are noisy and do not establish systematic miscalibration or perfect calibration:

| Mean predicted bad probability | Observed bad frequency | Applicants |
|---:|---:|---:|
| 0.023 | 0.040 | 25 |
| 0.054 | 0.120 | 25 |
| 0.094 | 0.160 | 25 |
| 0.131 | 0.040 | 25 |
| 0.191 | 0.160 | 25 |
| 0.279 | 0.200 | 25 |
| 0.389 | 0.400 | 25 |
| 0.495 | 0.560 | 25 |
| 0.632 | 0.560 | 25 |
| 0.807 | 0.760 | 25 |

For a small descriptive error check, the app reports held-out classification
errors and Brier scores by `purpose`, including only categories with at least
20 cases. At the unchanged 0.50 threshold, the qualifying groups are:

| Purpose code | Applicants | Observed bad rate | Brier score | False positives | False negatives |
|---|---:|---:|---:|---:|---:|
| A40 | 55 | 0.473 | 0.193 | 2 | 14 |
| A41 | 22 | 0.091 | 0.109 | 2 | 2 |
| A42 | 43 | 0.326 | 0.159 | 5 | 5 |
| A43 | 79 | 0.215 | 0.124 | 5 | 8 |
| A49 | 21 | 0.381 | 0.170 | 3 | 3 |

This exploratory view describes model errors across five purposes; it is not a
fairness analysis and does not support claims about protected groups or lending
impact.

All three Brier scores and the diagram use the fixed holdout; the prevalence
reference probability itself is computed from the training labels. No model,
threshold, or calibration parameter was tuned on holdout results. The holdout is
only 250 records from a small historical dataset. Reliability bins and subgroup
rates have substantial sampling uncertainty; Brier score summarizes probability
error but does not by itself isolate calibration from discrimination or
establish performance on a current applicant population.

## Limitations

This is a small historical benchmark dataset, not a representative or current lending population. It has class imbalance (70% good, 30% bad), and the current threshold-0.50 recall is imperfect: the model misses some bad-risk cases. A model score can encode the dataset's biases and does not establish causal risk or fairness. Coefficients describe the fitted model, not causal effects. The dashboard is educational: probabilities and classifications are estimates, not actual lending decisions, eligibility determinations, or advice. Performance on the held-out split does not guarantee future performance.
