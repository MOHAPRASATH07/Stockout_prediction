"""Subgroup Robustness and Stress-Testing Module.

Evaluates the trained stockout risk classifier under diverse operational regimes:
1. Normal Demand Days (sales within 20th-80th percentiles)
2. Demand Surge Days (sales in top 20th percentile)
3. Promotional Activity Days (activity_flag == 1 or discount < 0.90)
4. Non-Promotional Standard Days
5. Chronic Stockout History (stockout_freq_7d >= 0.50)
6. Clean Inventory History (stockout_freq_7d <= 0.10)
7. Extreme Weather Conditions (precpt > 0 or high wind)

Calculates PR-AUC, Recall, and F1 across each operating condition to identify degradation modes.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.target import add_target_variable
from src.split import temporal_split
from src.feature_engineering import prepare_feature_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight"})

def run_robustness_experiments(
    model_path: str = "models/best_model.pkl",
    fallback_path: str = "models/xgboost_model.joblib",
    output_csv: str = "results/metrics/robustness_results.csv",
    output_figure: str = "results/figures/16_robustness_subgroup_performance.png"
) -> pd.DataFrame:
    """Evaluate subgroup performance across real operating conditions."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_figure), exist_ok=True)

    # 1. Load test features
    df = load_data(use_sample=True)
    df = add_target_variable(df, "next_day")
    train_df, val_df, test_df, _ = temporal_split(df)
    train_f, val_f, test_f, feature_cols = prepare_feature_pipeline(train_df, val_df, test_df)

    active_path = model_path if os.path.exists(model_path) else fallback_path
    logger.info(f"Loading trained model from {active_path} for robustness audit...")
    model = joblib.load(active_path)

    X_test = test_f[feature_cols].fillna(0)
    y_test = test_f["target_stockout"].values
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    test_eval = test_f.copy()
    test_eval["y_true"] = y_test
    test_eval["y_prob"] = probs
    test_eval["y_pred"] = preds

    # Define operational condition masks
    p80_sales = test_eval["sale_amount"].quantile(0.80)
    p20_sales = test_eval["sale_amount"].quantile(0.20)

    subgroups = {
        "Overall Test Baseline": test_eval,
        "Normal Demand (20-80th pct)": test_eval[(test_eval["sale_amount"] >= p20_sales) & (test_eval["sale_amount"] <= p80_sales)],
        "High Demand Surge (Top 20%)": test_eval[test_eval["sale_amount"] > p80_sales],
        "Promotional Event Active": test_eval[(test_eval["activity_flag"] == 1) | (test_eval["discount"] < 0.90)],
        "Standard Full Price": test_eval[(test_eval["activity_flag"] == 0) & (test_eval["discount"] >= 0.99)],
        "Chronic Stockout History (>=50%)": test_eval[test_eval["stockout_freq_7d"] >= 0.50],
        "Stable In-Stock History (<=10%)": test_eval[test_eval["stockout_freq_7d"] <= 0.10],
        "Precipitation / Bad Weather": test_eval[test_eval["precpt"] > 0]
    }

    records = []
    for name, sub in subgroups.items():
        if len(sub) == 0:
            continue
        y_sub_true = sub["y_true"].values
        y_sub_prob = sub["y_prob"].values
        y_sub_pred = sub["y_pred"].values

        rec = recall_score(y_sub_true, y_sub_pred, zero_division=0)
        f1 = f1_score(y_sub_true, y_sub_pred, zero_division=0)
        try:
            pr_auc = average_precision_score(y_sub_true, y_sub_prob)
        except Exception:
            pr_auc = np.nan

        records.append({
            "Operating Condition": name,
            "Sample Count": len(sub),
            "Positive Rate (%)": round(float(y_sub_true.mean() * 100), 2),
            "PR-AUC": round(float(pr_auc), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4)
        })

    rob_df = pd.DataFrame(records)
    rob_df.to_csv(output_csv, index=False)
    logger.info(f"Robustness Results Saved to {output_csv}:\n{rob_df.to_string(index=False)}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df = rob_df[rob_df["Operating Condition"] != "Overall Test Baseline"].copy()
    
    y_pos = np.arange(len(plot_df))
    bar_width = 0.35
    
    ax.barh(y_pos - bar_width/2, plot_df["PR-AUC"], height=bar_width, label="PR-AUC", color="#2c7fb8")
    ax.barh(y_pos + bar_width/2, plot_df["F1-Score"], height=bar_width, label="F1-Score", color="#f03b20")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Operating Condition"])
    ax.set_xlabel("Metric Value")
    ax.set_title("Model Robustness and Performance Stability across Retail Regimes")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_figure)
    plt.close()
    logger.info(f"Saved robustness figure to {output_figure}")

    return rob_df

if __name__ == "__main__":
    run_robustness_experiments()
