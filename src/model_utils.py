from datetime import datetime
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def create_one_hot_encoder():
    """Create a OneHotEncoder that works across common scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def create_preprocessor(numeric_columns, categorical_columns):
    """Create preprocessing steps for numeric and categorical data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", create_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def create_classification_models(random_state: int):
    """Return the classification models used in the experiment."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
        "Random Forest Classifier": RandomForestClassifier(random_state=random_state),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            random_state=random_state
        ),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
    }


def create_regression_models(random_state: int):
    """Return the regression models used in the experiment."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(random_state=random_state),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=random_state
        ),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
    }


def train_models(models, X_train, y_train, numeric_columns, categorical_columns):
    """Train each model inside the same preprocessing pipeline."""
    trained_models = {}
    failed_models = {}

    for model_name, model in models.items():
        preprocessor = create_preprocessor(numeric_columns, categorical_columns)

        # The pipeline prevents data leakage because preprocessing is learned
        # only from the training data during fit.
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)
            trained_models[model_name] = pipeline
        except Exception as error:
            failed_models[model_name] = str(error)

    return trained_models, failed_models


def save_best_model(
    best_model,
    metadata: dict,
    model_path: str = "outputs/best_model.joblib",
    metadata_path: str = "outputs/model_metadata.json",
):
    """Save the best trained pipeline and its metadata."""
    import json

    Path("outputs").mkdir(exist_ok=True)
    metadata["training_date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    joblib.dump(best_model, model_path)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return model_path, metadata_path
