import pandas as pd


def build_prediction_input(input_values: dict) -> pd.DataFrame:
    """Convert user-entered feature values into one prediction row."""
    return pd.DataFrame([input_values])


def predict_single_row(model, input_values: dict):
    """Make one prediction using the trained best model pipeline."""
    input_df = build_prediction_input(input_values)
    return model.predict(input_df)[0]
