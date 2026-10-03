"""Advanced XGBoost Model for Stockout Risk Prediction.

Implements gradient-boosted decision trees with:
- Exact class imbalance weighting (scale_pos_weight)
- Time-aware validation early stopping
- PR-AUC and logloss evaluation
- Complete model serialization and metadata logging
"""

import os
import sys
import json
import logging
from typing import Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.target import add_target_variable
from src.split import temporal_split
from src.feature_engineering import prepare_feature_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute detailed evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4)
    }

def train_xgboost(
    params: dict = None,
    output_model_path: str = "models/xgboost_model.joblib",
    output_metadata_path: str = "results/metrics/xgboost_metadata.json"
) -> Tuple[xgb.XGBClassifier, dict]:
    """Train XGBoost model with early stopping on temporal validation split."""
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Load data and features
    df = load_data(use_sample=True)
    df = add_target_variable(df, "next_day")
    train_df, val_df, test_df, _ = temporal_split(df)
    train_f, val_f, test_f, feature_cols = prepare_feature_pipeline(train_df, val_df, test_df)

    X_train, y_train = train_f[feature_cols].fillna(0), train_f["target_stockout"]
    X_val, y_val = val_f[feature_cols].fillna(0), val_f["target_stockout"]
    X_test, y_test = test_f[feature_cols].fillna(0), test_f["target_stockout"]

    # Calculate exact class balance ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    logger.info(f"Class ratio: Negative={neg_count:,}, Positive={pos_count:,} -> scale_pos_weight={scale_pos_weight:.3f}")

    default_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": 42,
        "eval_metric": ["logloss", "aucpr"],
        "early_stopping_rounds": 30,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)

    # Instantiate XGBClassifier
    model = xgb.XGBClassifier(**default_params)
    logger.info("Fitting XGBoost Classifier with temporal validation early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )

    best_iteration = model.best_iteration if hasattr(model, "best_iteration") else model.n_estimators
    logger.info(f"Optimal iteration reached: {best_iteration}")

    # Predict probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_metrics(y_val.values, val_probs)
    test_metrics = evaluate_metrics(y_test.values, test_probs)

    logger.info(f"Validation Metrics: {val_metrics}")
    logger.info(f"Test Metrics: {test_metrics}")

    # Save model and predictions
    joblib.dump(model, output_model_path)
    
    # Save test predictions for error analysis & ROC/PR curves
    pred_df = test_f[["store_id", "product_id", "dt", "target_stockout"]].copy()
    pred_df["stockout_prob"] = test_probs
    pred_df["predicted_label"] = (test_probs >= 0.5).astype(int)
    os.makedirs("results/predictions", exist_ok=True)
    pred_df.to_parquet("results/predictions/xgboost_test_predictions.parquet", index=False)

    metadata = {
        "model_name": "XGBoost Classifier",
        "best_iteration": int(best_iteration),
        "hyperparameters": {k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v)) for k, v in default_params.items()},
        "feature_count": len(feature_cols),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    }

    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved to {output_model_path}")
    logger.info(f"Metadata saved to {output_metadata_path}")
    return model, metadata

if __name__ == "__main__":
    model, meta = train_xgboost()
