import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_target_distribution(y, task_type: str):
    """Create a target distribution chart."""
    fig, ax = plt.subplots(figsize=(7, 4))

    if task_type == "Classification":
        y.value_counts().plot(kind="bar", ax=ax, color="#4C78A8")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
    else:
        ax.hist(y, bins=20, color="#4C78A8", edgecolor="white")
        ax.set_xlabel("Target value")
        ax.set_ylabel("Frequency")

    ax.set_title("Target Distribution")
    fig.tight_layout()
    return fig


def plot_leaderboard(leaderboard, task_type: str):
    """Create a model leaderboard bar chart."""
    metric = "F1-score weighted" if task_type == "Classification" else "R2 score"
    sorted_scores = leaderboard.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sorted_scores["Model"], sorted_scores[metric], color="#59A14F")
    ax.set_xlabel(metric)
    ax.set_title("Model Leaderboard")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(model, X_test, y_test):
    """Create a confusion matrix for the best classification model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(y_test, predictions):
    """Create a predicted vs actual chart for regression."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, predictions, alpha=0.75, color="#F28E2B")

    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    ax.plot([min_value, max_value], [min_value, max_value], color="#333333")

    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Predicted vs Actual")
    fig.tight_layout()
    return fig
