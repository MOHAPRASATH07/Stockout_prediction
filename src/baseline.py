"""Baseline Models for Stockout Risk Prediction.

Implements three foundational baselines:
1. DummyClassifier (prior/majority class)
2. Logistic Regression (StandardScaler + L2 penalty)
3. Decision Tree Classifier (interpretable tree benchmark)

Evaluates on identical chronological splits using PR-AUC, ROC-AUC, F1, Recall, Precision, and Accuracy.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.target import add_target_variable
from src.split import temporal_split
from src.feature_engineering import prepare_feature_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str, split_name: str) -> dict:
    """Calculate all essential classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Handle constant predictions in dummy
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = y_true.mean()

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4)
    }

def train_baselines(output_models_dir: str = "models/baseline", output_metrics_path: str = "results/metrics/baseline_results.csv") -> pd.DataFrame:
    """Train and evaluate baseline models."""
    os.makedirs(output_models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Load data and prepare features
    df = load_data(use_sample=True)
    df = add_target_variable(df, "next_day")
    train_df, val_df, test_df, _ = temporal_split(df)
    train_f, val_f, test_f, feature_cols = prepare_feature_pipeline(train_df, val_df, test_df)

    X_train, y_train = train_f[feature_cols].fillna(0), train_f["target_stockout"]
    X_val, y_val = val_f[feature_cols].fillna(0), val_f["target_stockout"]
    X_test, y_test = test_f[feature_cols].fillna(0), test_f["target_stockout"]

    models = {
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
    }

    results = []
    test_conf_matrices = {}

    for name, model in models.items():
        logger.info(f"Training baseline: {name} ...")
        model.fit(X_train, y_train)

        # Validation evaluation
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else val_pred.astype(float)
        results.append(evaluate_predictions(y_val.values, val_pred, val_prob, name, "Validation"))

        # Test evaluation
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else test_pred.astype(float)
        results.append(evaluate_predictions(y_test.values, test_pred, test_prob, name, "Test"))

        test_conf_matrices[name] = confusion_matrix(y_test.values, test_pred)

        # Save model
        save_path = os.path.join(output_models_dir, f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, save_path)
        logger.info(f"Saved {name} to {save_path}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_metrics_path, index=False)
    logger.info(f"Baseline Results:\n{results_df.to_string(index=False)}")

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (name, cm) in zip(axes, test_conf_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["In-Stock (0)", "Stockout (1)"],
                    yticklabels=["In-Stock (0)", "Stockout (1)"])
        ax.set_title(f"{name} (Test Set)")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = "results/figures/baseline_confusion_matrices.png"
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Saved baseline confusion matrices to {cm_path}")

    return results_df

if __name__ == "__main__":
    df_res = train_baselines()
