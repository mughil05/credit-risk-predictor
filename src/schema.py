"""Schema for the original UCI Statlog German Credit data."""

RAW_FEATURES = [
    "checking_account_status", "duration_months", "credit_history",
    "purpose", "credit_amount", "savings_account_bonds",
    "employment_since", "installment_rate_percent", "personal_status_sex",
    "other_debtors_guarantors", "residence_since_years",
    "property", "age_years", "other_installment_plans", "housing",
    "existing_credits_count", "job", "dependents_count", "telephone",
    "foreign_worker",
]
ENGINEERED_FEATURES = ["credit_amount_per_month"]
FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
TARGET = "credit_risk"
NUMERIC_FEATURES = [
    "duration_months", "credit_amount", "installment_rate_percent",
    "residence_since_years", "age_years", "existing_credits_count",
    "dependents_count", "credit_amount_per_month",
]
CATEGORICAL_FEATURES = [name for name in RAW_FEATURES if name not in NUMERIC_FEATURES]

# The raw file uses 1 = good and 2 = bad. Our modeling convention is 1 = bad.
TARGET_MAPPING = {1: 0, 2: 1}

