#!/usr/bin/env python3
"""CLI script to train the EC price prediction model.

For the full appreciation model with Optuna HPO, run:
    python notebooks/02_appreciation_model.py

This script provides a simpler alternative for quick retraining.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sqlalchemy import text

from src.config import get_settings
from src.database import get_db
from src.features.engineering import build_features
from src.model.train import train_model
from src.model.experiment import get_best_run


def main() -> None:
    print("Loading transactions from Postgres...")
    with get_db() as db:
        df = pd.read_sql(text("SELECT * FROM ura_transactions"), db.bind)

    print(f"Total transactions: {len(df)}")

    print("Building features...")
    features_df = build_features(df)
    print(f"EC feature rows: {len(features_df)}")

    if features_df.empty:
        print("ERROR: No EC transactions found. Run ingestion first.")
        sys.exit(1)

    print("\nTraining XGBoost...")
    meta = train_model(features_df, artifact_dir="./artifacts", algorithm="xgboost")

    m = meta["metrics"]
    print(f"\n{'='*50}")
    print(f"Run {meta['run_id']}:")
    print(f"  R²:   {m['r2']:.4f}")
    print(f"  MAE:  S${m['mae']:,.0f}")
    print(f"  MAPE: {m['mape']:.2f}%")
    print(f"\nModel saved to ./artifacts/model.joblib")
    print(f"\nFor the full appreciation model with HPO, run:")
    print(f"  python notebooks/02_appreciation_model.py")


if __name__ == "__main__":
    main()
