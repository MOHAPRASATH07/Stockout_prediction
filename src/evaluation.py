"""Rigorous Model Comparison and Evaluation Module.

Evaluates all trained candidate models on the held-out Test set:
1. Dummy Classifier
2. Logistic Regression
3. Decision Tree
4. XGBoost (Tuned)

Generates:
- final_model_comparison.csv
- ROC curves (roc_curves.png)
- Precision-Recall curves (pr_curves.png)
- Reliability / Calibration curves (calibration_curves.png)
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.target import add_target_variable
from src.split import temporal_split
from src.feature_engineering import prepare_feature_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight"})

def run_model_comparison(output_dir: str = "results/figures", metrics_dir: str = "results/metrics") -> pd.DataFrame:
    """Load all models, evaluate on held-out test split, and generate comparison artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Load test data
    df = load_data(use_sample=True)
    df = add_target_variable(df, "next_day")
    train_df, val_df, test_df, _ = temporal_split(df)
    train_f, val_f, test_f, feature_cols = prepare_feature_pipeline(train_df, val_df, test_df)

    X_test = test_f[feature_cols].fillna(0)
    y_test = test_f["target_stockout"].values

    # 2. Candidate model filepaths
    model_paths = {
        "Dummy (Stratified)": "models/baseline/dummy_(stratified).joblib",
        "Logistic Regression": "models/baseline/logistic_regression.joblib",
        "Decision Tree": "models/baseline/decision_tree.joblib",
        "XGBoost (Tuned)": "models/best_model.pkl" if os.path.exists("models/best_model.pkl") else "models/xgboost_model.joblib"
    }

    comparison_records = []
    model_preds = {}

    colors = {
        "Dummy (Stratified)": "#999999",
        "Logistic Regression": "#386cb0",
        "Decision Tree": "#7570b3",
        "XGBoost (Tuned)": "#d95f02"
    }

    for name, path in model_paths.items():
        if not os.path.exists(path):
            logger.warning(f"Model path {path} not found. Skipping {name}.")
            continue
        logger.info(f"Evaluating {name} on held-out Test set...")
        model = joblib.load(path)
        
        # Predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test).astype(float)
        preds = (probs >= 0.5).astype(int)

        model_preds[name] = {"probs": probs, "preds": preds}

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        comparison_records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC (Primary)": round(pr_auc, 4),
            "Brier Score": round(brier, 4)
        })

    # Save CSV comparison
    comp_df = pd.DataFrame(comparison_records).sort_values("PR-AUC (Primary)", ascending=False)
    csv_path = os.path.join(metrics_dir, "final_model_comparison.csv")
    comp_df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison to {csv_path}:\n{comp_df.to_string(index=False)}")

    # 3. Generate ROC Curves
    plt.figure(figsize=(8, 6))
    for name, pred in model_preds.items():
        fpr, tpr, _ = roc_curve(y_test, pred["probs"])
        auc_val = roc_auc_score(y_test, pred["probs"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors.get(name, "black"), linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (0.500)")
    plt.title("Receiver Operating Characteristic (ROC) Curves - Test Set")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "11_roc_curves.png"))
    plt.close()

    # 4. Generate Precision-Recall Curves
    baseline_pr = y_test.mean()
    plt.figure(figsize=(8, 6))
    for name, pred in model_preds.items():
        precision, recall, _ = precision_recall_curve(y_test, pred["probs"])
        pr_auc_val = average_precision_score(y_test, pred["probs"])
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", color=colors.get(name, "black"), linewidth=2)
    plt.axhline(baseline_pr, color="black", linestyle="--", alpha=0.6, label=f"No Skill Prevalence ({baseline_pr:.3f})")
    plt.title("Precision-Recall (PR) Curves - Test Set (Primary Metric)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "12_pr_curves.png"))
    plt.close()

    # 5. Calibration / Reliability Curves
    plt.figure(figsize=(8, 6))
    for name, pred in model_preds.items():
        if name == "Dummy (Stratified)":
            continue
        prob_true, prob_pred = calibration_curve(y_test, pred["probs"], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=f"{name}", color=colors.get(name, "black"), linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.title("Probability Calibration (Reliability Curves)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "13_calibration_curves.png"))
    plt.close()

    logger.info("Model evaluation and publication curve generation complete.")
    return comp_df

if __name__ == "__main__":
    run_model_comparison()
