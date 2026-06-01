from pathlib import Path
from typing import Optional

import pandas as pd


SAMPLE_CLASSIFICATION_PATH = Path("data/sample_classification.csv")
SAMPLE_REGRESSION_PATH = Path("data/sample_regression.csv")


def load_csv(file_or_path) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_or_path)


def load_sample_dataset(sample_type: str) -> pd.DataFrame:
    """Load one of the project sample datasets."""
    if sample_type == "Sample classification dataset":
        return load_csv(SAMPLE_CLASSIFICATION_PATH)
    return load_csv(SAMPLE_REGRESSION_PATH)


def get_dataset_summary(df: pd.DataFrame, target_column: Optional[str] = None) -> dict:
    """Collect simple dataset facts for display in the app."""
    missing_values = df.isna().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).reset_index().rename(
            columns={"index": "Column", 0: "Data Type"}
        ),
        "missing_values": missing_values,
        "duplicated_rows": int(df.duplicated().sum()),
        "target_missing_count": int(df[target_column].isna().sum())
        if target_column in df.columns
        else 0,
    }


def clean_target_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Remove rows where the target value is missing."""
    return df.dropna(subset=[target_column]).copy()


def split_features_target(df: pd.DataFrame, target_column: str):
    """
    Separate a dataset into X and y.

    X means input features: the columns the model uses to learn patterns.
    y means prediction target: the column the model is trying to predict.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_feature_columns(df: pd.DataFrame, target_column: str):
    """Detect numeric and categorical feature columns."""
    feature_df = df.drop(columns=[target_column])

    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_columns, categorical_columns
