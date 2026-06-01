def dataframe_to_markdown_table(df):
    """Create a markdown table without extra dependencies."""
    headers = df.columns.tolist()
    rows = df.astype(str).values.tolist()

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]

    return "\n".join([header_line, separator_line] + row_lines)


def generate_markdown_report(
    dataset_shape,
    target_column,
    task_type,
    test_size,
    random_state,
    numeric_columns,
    categorical_columns,
    leaderboard,
    best_model_name,
):
    """Create a simple markdown experiment report."""
    metric_explanation = (
        "Classification best model is selected by weighted F1-score."
        if task_type == "Classification"
        else "Regression best model is selected by R2 score."
    )

    leaderboard_markdown = dataframe_to_markdown_table(leaderboard)

    return f"""# AutoML Experiment Lab Report

## Project Name
AutoML Experiment Lab

## Dataset
- Shape: {dataset_shape[0]} rows x {dataset_shape[1]} columns
- Target column: {target_column}
- Task type: {task_type}
- Test size: {test_size}
- Random state: {random_state}

## Feature Columns
- Numeric columns: {", ".join(numeric_columns) if numeric_columns else "None"}
- Categorical columns: {", ".join(categorical_columns) if categorical_columns else "None"}

## Model Leaderboard
{leaderboard_markdown}

## Best Model
{best_model_name}

## Selected Metric
{metric_explanation}

## Short Conclusion
The best model for this experiment was {best_model_name}. This result is based on the selected metric and the test data split used in the app.
"""
