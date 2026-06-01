import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def evaluate_classification_model(model, X_test, y_test):
    """Calculate classification scores for a trained model."""
    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1-score weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }


def evaluate_regression_model(model, X_test, y_test):
    """Calculate regression scores for a trained model."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mse),
        "R2 score": r2_score(y_test, predictions),
    }


def build_leaderboard(trained_models, X_test, y_test, task_type: str):
    """Evaluate all trained models and return a sorted leaderboard."""
    results = []
    failed_evaluations = {}

    for model_name, model in trained_models.items():
        try:
            if task_type == "Classification":
                scores = evaluate_classification_model(model, X_test, y_test)
            else:
                scores = evaluate_regression_model(model, X_test, y_test)

            row = {"Model": model_name}
            row.update({metric: round(value, 4) for metric, value in scores.items()})
            results.append(row)
        except Exception as error:
            failed_evaluations[model_name] = str(error)

    leaderboard = pd.DataFrame(results)

    if leaderboard.empty:
        return leaderboard, failed_evaluations

    sort_column = (
        "F1-score weighted" if task_type == "Classification" else "R2 score"
    )

    leaderboard = leaderboard.sort_values(
        by=sort_column,
        ascending=False,
    ).reset_index(drop=True)

    return leaderboard, failed_evaluations
