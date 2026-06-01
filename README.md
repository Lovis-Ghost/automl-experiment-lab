# AutoML Experiment Lab

## Project Overview

AutoML Experiment Lab is a beginner-friendly machine learning experiment platform built with Streamlit and scikit-learn. It lets users upload a CSV file or use built-in sample datasets, choose a target column, select a classification or regression task, train multiple models, compare results, save the best model, download a markdown report, and test a single-row prediction.

This is an AutoML-style project, but it does not use a full AutoML library. The workflow is built manually with scikit-learn pipelines so the data preprocessing, model training, evaluation, and model saving steps are easy to understand.

## Why I Built This Project

I built this project to learn how AutoML-style tools work behind the scenes. Instead of hiding the process, the app walks through the core machine learning workflow: loading data, selecting features and a target, preprocessing numeric and categorical columns, training multiple models, evaluating performance, and exporting the best model.

The goal is to make machine learning experimentation easier for beginners while still using clean, practical project structure suitable for GitHub and portfolio review.

## Key Features

- Upload a custom CSV dataset
- Use a sample classification dataset or sample regression dataset
- Select the target column to predict
- Choose Classification or Regression
- Automatically detect numeric and categorical features
- Preprocess data with scikit-learn pipelines
- Train multiple machine learning models
- Compare models in a leaderboard
- Visualize target distribution and model performance
- Save the best model to `outputs/best_model.joblib`
- Save metadata to `outputs/model_metadata.json`
- Download a markdown experiment report
- Make a simple single-row prediction with the best model
- Avoid stale results when experiment settings change

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Joblib

## Project Structure

```text
AutoML-Experiment-Lab/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── sample_classification.csv
│   └── sample_regression.csv
├── outputs/
│   ├── best_model.joblib
│   └── model_metadata.json
└── src/
    ├── data_utils.py
    ├── evaluation.py
    ├── model_utils.py
    ├── prediction_utils.py
    ├── report_utils.py
    └── visualization.py
```

## How to Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## How to Use the App

1. Open the app in your browser.
2. Upload a CSV file or use one of the sample datasets.
3. Select the target column. This is the value the model will predict.
4. Choose Classification for category prediction or Regression for numeric prediction.
5. Adjust test size and random state if needed.
6. Click Run Experiment.
7. Review the model leaderboard and charts.
8. Check the best model details.
9. Download the markdown report.
10. Try a single-row prediction using the trained best model.

## Machine Learning Workflow

1. Load the dataset from a CSV upload or sample file.
2. Remove rows where the target column is missing.
3. Split the dataset into features `X` and target `y`.
4. Split the data into training and test sets.
5. Detect numeric and categorical feature columns.
6. Build preprocessing pipelines for missing values, scaling, and one-hot encoding.
7. Train several classification or regression models.
8. Evaluate each model on the test set.
9. Rank models in a leaderboard.
10. Save the best trained pipeline and metadata.
11. Use the best model for a single-row prediction.

## Example Use Cases

- Compare churn prediction models on a customer dataset.
- Predict student final scores from study and attendance data.
- Quickly test whether a CSV dataset is better suited for classification or regression.
- Build a portfolio demo showing practical scikit-learn pipeline usage.
- Generate a simple experiment report for a machine learning notebook or project write-up.

## Screenshots

Add screenshots of the running Streamlit app here after testing locally.

Suggested screenshots:

- Dataset Overview
- Feature Detection
- Model Leaderboard
- Best Model Details
- Single Row Prediction

## What I Learned

- How to build an interactive machine learning app with Streamlit
- How to structure a beginner-friendly Python ML project
- How to use scikit-learn `Pipeline` and `ColumnTransformer`
- How to preprocess numeric and categorical data safely
- How to compare classification and regression models
- How to save trained models with Joblib
- How to generate a markdown experiment report
- How to manage Streamlit `session_state` so old results do not appear after settings change

## Future Improvements

- Add optional cross-validation
- Add more model choices
- Add simple feature importance charts
- Add batch prediction for uploaded CSV files
- Add clearer data quality warnings
- Add optional saved-model loading

## Testing Checklist

- [ ] Run sample classification dataset with target `churn`
- [ ] Run sample regression dataset with target `final_score`
- [ ] Upload custom CSV dataset
- [ ] Check model leaderboard
- [ ] Check best model saving
- [ ] Download experiment report
- [ ] Test single-row prediction

## Resume Bullet Point

Built a beginner-friendly AutoML-style experiment platform using Streamlit and scikit-learn, supporting CSV upload, automated preprocessing with pipelines, classification and regression model comparison, performance visualization, markdown report generation, single-row prediction, and best model export.
