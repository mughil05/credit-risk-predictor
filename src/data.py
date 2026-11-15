from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from src.schema import FEATURES, RAW_FEATURES, TARGET


def create_spark_session():
    return (SparkSession.builder.master("local[*]")
            .appName("GermanCreditPreprocessing")
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def load_and_preprocess(path: str | Path) -> pd.DataFrame:
    """Read raw whitespace-delimited records with Spark and return cleaned pandas data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See README for the UCI download.")
    spark = create_spark_session()
    try:
        raw = spark.read.text(str(path))
        fields = F.split(F.trim(F.col("value")), r"\s+")
        parsed = raw.select(*[
            fields.getItem(i).alias(name) for i, name in enumerate(RAW_FEATURES)
        ], fields.getItem(20).cast("int").alias("raw_target"))
        # Empty strings are represented as null so downstream imputers can handle them.
        cleaned = parsed.select(*[
            F.when(F.trim(F.col(c)) == "", None).otherwise(F.trim(F.col(c))).alias(c)
            for c in RAW_FEATURES
        ], F.col("raw_target"))
        for c in ["duration_months", "credit_amount", "installment_rate_percent",
                  "residence_since_years", "age_years", "existing_credits_count",
                  "dependents_count"]:
            cleaned = cleaned.withColumn(c, F.col(c).cast("double"))
        cleaned = cleaned.withColumn(
            "credit_amount_per_month",
            F.col("credit_amount") / F.col("duration_months"),
        )
        cleaned = cleaned.withColumn(
            TARGET,
            F.when(F.col("raw_target") == 1, 0)
             .when(F.col("raw_target") == 2, 1)
             .otherwise(None),
        ).drop("raw_target")
        # Drop malformed rows, preserving every valid original observation.
        cleaned = cleaned.filter(F.col(TARGET).isNotNull())
        # Avoid Spark's pandas bridge here (it relies on distutils, removed in
        # Python 3.12); collect the small 1,000-row benchmark into pandas.
        return pd.DataFrame([row.asDict() for row in cleaned.collect()],
                            columns=FEATURES + [TARGET])
    finally:
        spark.stop()

