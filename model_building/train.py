"""
Model Training Module
---------------------
Trains an XGBoost classification model using prepared dataset splits,
logs metrics to MLflow, and uploads the trained model to Hugging Face Hub.

Authentication:
- Uses environment variable `HF_TOKEN`
"""

# ============================================================
# Imports
# ============================================================
import os
from typing import Optional

import joblib
import mlflow
import pandas as pd
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from huggingface_hub import HfApi


# ============================================================
# Configuration
# ============================================================
MODEL_REPO_ID = "praveenchugh/tourism-package-prediction-model"
MODEL_FILE = "best_mlops_tourism_model.joblib"

MLFLOW_TRACKING_URI = "http://localhost:5001"
MLFLOW_EXPERIMENT = "mlops-training-experiment"

CLASSIFICATION_THRESHOLD = 0.45

NUMERIC_FEATURES = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
]

CATEGORICAL_FEATURES = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
    "Passport",
    "OwnCar",
]


# ============================================================
# Authentication
# ============================================================
def get_hf_token() -> str:
    """Retrieve Hugging Face token from environment variable."""
    token: Optional[str] = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set.")

    return token


def get_hf_client() -> HfApi:
    """Create authenticated Hugging Face API client."""
    return HfApi(token=get_hf_token())


# ============================================================
# Data Loading
# ============================================================
def load_splits():
    """Load prepared train/test splits from local CSV files."""
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()

    print("Dataset splits loaded successfully.")
    return Xtrain, Xtest, ytrain, ytest


# ============================================================
# Model Pipeline
# ============================================================
def build_pipeline(scale_pos_weight: float):
    """Create preprocessing + XGBoost pipeline."""
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return make_pipeline(preprocessor, model)


# ============================================================
# Training + MLflow Logging
# ============================================================
def train_and_log_model(Xtrain, Xtest, ytrain, ytest) -> str:
    """Train model, log metrics to MLflow, and save artifact."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
    pipeline = build_pipeline(scale_pos_weight)

    with mlflow.start_run():

        pipeline.fit(Xtrain, ytrain)

        y_train_pred = (pipeline.predict_proba(Xtrain)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)
        y_test_pred = (pipeline.predict_proba(Xtest)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)

        train_report = classification_report(ytrain, y_train_pred, output_dict=True)
        test_report = classification_report(ytest, y_test_pred, output_dict=True)

        mlflow.log_metrics({
            "train_accuracy": train_report["accuracy"],
            "train_f1": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_f1": test_report["1"]["f1-score"],
        })

        joblib.dump(pipeline, MODEL_FILE)
        mlflow.log_artifact(MODEL_FILE, artifact_path="model")

    print("Model training and MLflow logging completed.")
    return MODEL_FILE


# ============================================================
# Upload Model to Hugging Face
# ============================================================
def upload_model(api: HfApi, model_path: str):
    """Upload trained model artifact to Hugging Face Hub."""
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
    )

    print("Model uploaded to Hugging Face successfully.")


# ============================================================
# Main Execution
# ============================================================
def main():
    api = get_hf_client()

    Xtrain, Xtest, ytrain, ytest = load_splits()
    model_path = train_and_log_model(Xtrain, Xtest, ytrain, ytest)
    upload_model(api, model_path)

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
