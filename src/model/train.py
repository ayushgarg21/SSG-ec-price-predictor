"""Model training pipeline for EC price prediction with experiment tracking.

This module provides the programmatic API. For the full advanced training pipeline
with Optuna HPO, multi-model comparison, and SHAP analysis, run:
    python notebooks/02_train.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.model.experiment import log_experiment

logger = structlog.get_logger()

# Base features (used for API serving)
FEATURE_COLUMNS: list[str] = [
    "district_num",
    "area",
    "floor_mid",
    "lease_commence_year",
    "years_from_launch",
    "remaining_lease",
    "sale_type_encoded",
    "segment_encoded",
]

# Advanced features (added by notebooks/02_train.py)
ADVANCED_FEATURE_COLUMNS: list[str] = [
    "district_num", "area", "floor_mid", "lease_commence_year",
    "years_from_launch", "remaining_lease", "sale_type_encoded", "segment_encoded",
    "district_median_psm", "district_volume", "project_median_psm",
    "project_total_units", "project_median_area",
    "area_quartile", "is_post_mop", "is_privatised", "quarter",
    "lease_age_bucket", "is_high_floor",
]

TARGET_COLUMN: str = "price"


def _build_pipeline(algorithm: str) -> Pipeline:
    """Construct a preprocessing + regressor pipeline."""
    if algorithm == "xgboost":
        regressor = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
        )
    elif algorithm == "gbr":
        regressor = GradientBoostingRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'xgboost' or 'gbr'.")

    return Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def train_model(
    df: pd.DataFrame,
    artifact_dir: str = "./artifacts",
    algorithm: str = "xgboost",
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and persist a price prediction model."""
    os.makedirs(artifact_dir, exist_ok=True)
    cols = feature_columns or FEATURE_COLUMNS

    X = df[cols].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = _build_pipeline(algorithm)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = _evaluate(y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    metrics["cv_r2_mean"] = float(cv_scores.mean())
    metrics["cv_r2_std"] = float(cv_scores.std())

    logger.info("model_trained", algorithm=algorithm, r2=metrics["r2"], mae=metrics["mae"])

    model_path = Path(artifact_dir) / "model.joblib"
    joblib.dump(model, model_path)

    params = model.named_steps["regressor"].get_params()
    run_id = log_experiment(
        algorithm=algorithm, parameters=params, metrics=metrics,
        feature_columns=cols, train_size=len(X_train), test_size=len(X_test),
        artifact_path=str(model_path),
    )

    metadata: dict[str, Any] = {
        "run_id": run_id, "algorithm": algorithm, "feature_columns": cols,
        "metrics": metrics, "train_size": len(X_train), "test_size": len(X_test),
        "parameters": params,
    }

    with open(Path(artifact_dir) / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


def load_model(model_path: str = "./artifacts/model.joblib") -> Pipeline:
    """Load a trained model from disk."""
    return joblib.load(model_path)
