from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_utils import (
    clean_target_rows,
    get_dataset_summary,
    get_feature_columns,
    load_csv,
    load_sample_dataset,
    split_features_target,
)
from src.evaluation import build_leaderboard
from src.model_utils import (
    create_classification_models,
    create_regression_models,
    save_best_model,
    train_models,
)
from src.prediction_utils import predict_single_row
from src.report_utils import generate_markdown_report
from src.visualization import (
    plot_confusion_matrix,
    plot_leaderboard,
    plot_predicted_vs_actual,
    plot_target_distribution,
)


st.set_page_config(page_title="AutoML Experiment Lab", layout="wide")


def main():
    st.title("AutoML Experiment Lab")
    st.write(
        "A beginner-friendly AutoML-style lab for uploading a CSV dataset, "
        "training several scikit-learn models, comparing results, saving the "
        "best model, and making simple predictions."
    )

    df = None
    load_error = None

    with st.sidebar:
        st.header("Experiment Settings")

        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
        sample_choice = st.selectbox(
            "Sample dataset",
            ["Sample classification dataset", "Sample regression dataset"],
        )

        try:
            if uploaded_file is not None:
                df = load_csv(uploaded_file)
            else:
                df = load_sample_dataset(sample_choice)
        except Exception as error:
            load_error = error

        if df is not None and not df.empty:
            target_options = [""] + df.columns.tolist()
            default_target = ""
            if uploaded_file is None and sample_choice == "Sample classification dataset":
                default_target = "churn"
            if uploaded_file is None and sample_choice == "Sample regression dataset":
                default_target = "final_score"
            default_index = (
                target_options.index(default_target)
                if default_target in target_options
                else 0
            )
            target_column = st.selectbox(
                "Target column", target_options, index=default_index
            )
            st.caption("The target column is the value the model will learn to predict.")
        else:
            target_column = ""

        default_task = (
            "Regression"
            if sample_choice == "Sample regression dataset" and uploaded_file is None
            else "Classification"
        )
        task_type = st.radio(
            "Task type",
            ["Classification", "Regression"],
            index=0 if default_task == "Classification" else 1,
        )
        st.caption(
            "Classification predicts a category, such as churn or not churn. "
            "Regression predicts a number, such as a score or price."
        )
        test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
        random_state = st.number_input(
            "Random state", min_value=0, max_value=9999, value=42, step=1
        )
        run_button = st.button("Run Experiment", type="primary")

    current_config = {
        "sample_choice": sample_choice,
        "uploaded_file_name": uploaded_file.name if uploaded_file is not None else None,
        "target_column": target_column,
        "task_type": task_type,
        "test_size": test_size,
        "random_state": int(random_state),
    }

    if load_error:
        st.error(
            "The CSV file could not be read. Please check that it is a valid "
            f"CSV file and try again. Details: {load_error}"
        )
        return

    if df is None or df.empty:
        st.error(
            "The dataset is empty. Please upload a CSV file that has at least "
            "one row and one column."
        )
        return

    show_dataset_overview(df, target_column)

    if not target_column:
        st.info(
            "Please select a target column in the sidebar. This is the column "
            "the model will try to predict."
        )
        return

    if run_button:
        run_experiment(
            df=df,
            target_column=target_column,
            task_type=task_type,
            test_size=test_size,
            random_state=int(random_state),
            config=current_config,
        )
    elif "experiment_results" in st.session_state:
        results = st.session_state["experiment_results"]
        stored_config = results.get("config")
        if stored_config == current_config:
            display_results(**results["display"])
            st.info(
                f"Best model is still available from the last run: {results['best_model_name']}"
            )
            show_prediction_section(
                results["best_model"],
                results["feature_df"],
                results["numeric_columns"],
                results["categorical_columns"],
                results["task_type"],
            )
        else:
            st.info("Settings changed. Please run the experiment again.")


def show_dataset_overview(df: pd.DataFrame, target_column: str):
    """Display beginner-friendly dataset information."""
    summary = get_dataset_summary(df, target_column)

    st.subheader("Dataset Overview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{summary['shape'][0]:,}")
    col2.metric("Columns", f"{summary['shape'][1]:,}")
    col3.metric("Duplicated Rows", f"{summary['duplicated_rows']:,}")

    with st.expander("Column Names", expanded=False):
        st.write(summary["columns"])

    col_left, col_right = st.columns(2)
    with col_left:
        st.write("Data types")
        st.dataframe(summary["data_types"], use_container_width=True, hide_index=True)
    with col_right:
        st.write("Missing value summary")
        st.dataframe(
            summary["missing_values"], use_container_width=True, hide_index=True
        )


def run_experiment(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    test_size: float,
    random_state: int,
    config: dict,
):
    """Run the full model comparison experiment."""
    if target_column not in df.columns:
        st.error("The selected target column does not exist in the dataset.")
        return

    target_missing_count = int(df[target_column].isna().sum())
    target_missing_rate = target_missing_count / len(df)
    if target_missing_rate > 0.5:
        st.error(
            "The target column is missing too many values. Choose another target "
            "or clean the dataset first."
        )
        return

    cleaned_df = clean_target_rows(df, target_column)
    if cleaned_df.empty:
        st.error("No rows remain after removing missing target values.")
        return

    # X means input features: columns used to make predictions.
    # y means prediction target: the column we want the model to predict.
    X, y = split_features_target(cleaned_df, target_column)

    if X.empty:
        st.error("There are no feature columns left after selecting the target.")
        return

    numeric_columns, categorical_columns = get_feature_columns(cleaned_df, target_column)
    if not numeric_columns and not categorical_columns:
        st.error("No usable numeric or categorical feature columns were found.")
        return

    validation_passed = validate_target(y, task_type)
    if not validation_passed:
        return

    with st.expander("Feature Detection", expanded=True):
        st.write("Numeric columns:", numeric_columns or "None")
        st.write("Categorical columns:", categorical_columns or "None")

    # A train/test split means training data for learning and test data as
    # exam data for checking how well the model performs on unseen rows.
    stratify_values = None
    if task_type == "Classification" and y.value_counts().min() >= 2:
        stratify_values = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_values,
        )
    except Exception as error:
        st.error(
            "The app could not split the dataset into training and test data. "
            "This can happen when the dataset is too small or a class has too "
            f"few rows. Details: {error}"
        )
        return

    models = (
        create_classification_models(random_state)
        if task_type == "Classification"
        else create_regression_models(random_state)
    )

    with st.spinner("Training models..."):
        trained_models, training_failures = train_models(
            models=models,
            X_train=X_train,
            y_train=y_train,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )
        leaderboard, evaluation_failures = build_leaderboard(
            trained_models, X_test, y_test, task_type
        )

    for model_name, message in training_failures.items():
        st.warning(
            f"{model_name} could not be trained, so it was skipped. "
            f"The other models will continue. Details: {message}"
        )

    for model_name, message in evaluation_failures.items():
        st.warning(
            f"{model_name} could not be evaluated, so it was skipped. "
            f"The other models will continue. Details: {message}"
        )

    if leaderboard.empty:
        st.error("No models could be trained and evaluated successfully.")
        return

    best_model_name = leaderboard.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_metrics = {
        metric: float(value)
        for metric, value in leaderboard.iloc[0].drop(labels=["Model"]).to_dict().items()
    }

    display_results(
        leaderboard=leaderboard,
        best_model_name=best_model_name,
        best_model=best_model,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type,
    )

    st.session_state["experiment_results"] = {
        "config": config,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "feature_df": X,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "task_type": task_type,
        "display": {
            "leaderboard": leaderboard,
            "best_model_name": best_model_name,
            "best_model": best_model,
            "X_test": X_test,
            "y_test": y_test,
            "task_type": task_type,
        },
    }

    metadata = {
        "task_type": task_type,
        "target_column": target_column,
        "best_model_name": best_model_name,
        "metrics": best_metrics,
        "feature_column_names": X.columns.tolist(),
    }
    model_path, metadata_path = save_best_model(best_model, metadata)
    st.success(f"Saved best model to {model_path}")
    st.caption(f"Saved metadata to {metadata_path}")

    st.subheader("Report Download")
    report = generate_markdown_report(
        dataset_shape=cleaned_df.shape,
        target_column=target_column,
        task_type=task_type,
        test_size=test_size,
        random_state=random_state,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        leaderboard=leaderboard,
        best_model_name=best_model_name,
    )
    st.download_button(
        "Download Experiment Report",
        data=report,
        file_name="automl_experiment_report.md",
        mime="text/markdown",
    )

    show_prediction_section(best_model, X, numeric_columns, categorical_columns, task_type)


def validate_target(y: pd.Series, task_type: str) -> bool:
    """Validate the target column for the chosen task type."""
    if task_type == "Classification":
        class_count = y.nunique(dropna=True)
        if class_count < 2:
            st.error(
                "Classification needs at least two different target classes. "
                "Please choose a target column with two or more categories."
            )
            return False
        if class_count > 20:
            st.warning(
                "This target has many unique values. It may not be a good "
                "classification target."
            )

    if task_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
        st.error(
            "Regression needs a numeric target column. Please choose a column "
            "that contains numbers, such as prices, scores, or amounts."
        )
        return False

    return True


def display_results(leaderboard, best_model_name, best_model, X_test, y_test, task_type):
    """Show leaderboard, charts, and best model details."""
    st.subheader("Model Leaderboard")
    st.write(
        "The leaderboard ranks each trained model by its test-set performance, "
        "so you can quickly compare which model worked best for this experiment."
    )
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    if task_type == "Classification":
        st.info("Classification best model is selected by weighted F1-score.")
    else:
        st.info("Regression best model is selected by R2 score.")

    st.success(f"Best model: {best_model_name}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(plot_target_distribution(y_test, task_type))
    with chart_col2:
        st.pyplot(plot_leaderboard(leaderboard, task_type))

    predictions = best_model.predict(X_test)

    if task_type == "Classification":
        st.subheader("Best Model Details")
        st.pyplot(plot_confusion_matrix(best_model, X_test, y_test))
        report_dict = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)
    else:
        st.subheader("Best Model Details")
        st.pyplot(plot_predicted_vs_actual(y_test, predictions))


def show_prediction_section(
    best_model,
    feature_df: pd.DataFrame,
    numeric_columns,
    categorical_columns,
    task_type: str,
):
    """Allow the user to enter one row and predict with the best model."""
    st.subheader("Single Row Prediction")
    st.write("Enter one row of feature values, then click Predict.")

    input_values = {}
    for column in feature_df.columns:
        if column in numeric_columns:
            median_value = feature_df[column].median()
            if pd.isna(median_value):
                median_value = 0
            input_values[column] = st.number_input(
                column,
                value=float(median_value),
                key=f"predict_{column}",
            )
        elif column in categorical_columns:
            options = sorted(feature_df[column].dropna().astype(str).unique().tolist())
            if not options:
                options = [""]
            input_values[column] = st.selectbox(
                column,
                options=options,
                key=f"predict_{column}",
            )

    if st.button("Predict"):
        try:
            prediction = predict_single_row(best_model, input_values)
            if task_type == "Classification":
                st.success(f"Predicted class: {prediction}")
            else:
                st.success(f"Predicted value: {float(prediction):.2f}")
        except Exception as error:
            st.error(f"Prediction failed: {error}")


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    main()
