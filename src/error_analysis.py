"""Error Analysis Module for Stockout Risk Prediction.

Analyzes False Positives (FP) and False Negatives (FN) across:
- Primary Category
- Promotional Activity (activity_flag)
- Discount Depth
- Sales Volume Tiers (Low, Medium, High)
- Historical Stockout History
- Demand Volatility Tiers

Produces granular slices, confusion profiles, and publication figures.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.target import add_target_variable
from src.split import temporal_split
from src.feature_engineering import prepare_feature_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight"})

def run_error_analysis(
    model_path: str = "models/best_model.pkl",
    fallback_model_path: str = "models/xgboost_model.joblib",
    output_csv_path: str = "results/metrics/error_analysis.csv",
    output_figures_dir: str = "results/figures"
) -> pd.DataFrame:
    """Perform slice-based error analysis on the held-out Test set."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(output_figures_dir, exist_ok=True)

    # 1. Load data & test features
    df = load_data(use_sample=True)
    df = add_target_variable(df, "next_day")
    train_df, val_df, test_df, _ = temporal_split(df)
    train_f, val_f, test_f, feature_cols = prepare_feature_pipeline(train_df, val_df, test_df)

    active_model_path = model_path if os.path.exists(model_path) else fallback_model_path
    logger.info(f"Loading trained model from {active_model_path} for error analysis...")
    model = joblib.load(active_model_path)

    X_test = test_f[feature_cols].fillna(0)
    y_test = test_f["target_stockout"].values
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    analysis_df = test_f.copy()
    analysis_df["true_label"] = y_test
    analysis_df["predicted_prob"] = probs
    analysis_df["predicted_label"] = preds

    # Categorize error types
    conditions = [
        (analysis_df["true_label"] == 1) & (analysis_df["predicted_label"] == 1),
        (analysis_df["true_label"] == 0) & (analysis_df["predicted_label"] == 0),
        (analysis_df["true_label"] == 0) & (analysis_df["predicted_label"] == 1),
        (analysis_df["true_label"] == 1) & (analysis_df["predicted_label"] == 0)
    ]
    choices = ["True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"]
    analysis_df["error_type"] = np.select(conditions, choices, default="Unknown")
    analysis_df["is_error"] = (analysis_df["true_label"] != analysis_df["predicted_label"]).astype(int)

    # Create diagnostic slices
    analysis_df["sales_tier"] = pd.qcut(analysis_df["sale_amount"], q=3, labels=["Low Demand", "Medium Demand", "High Demand"], duplicates="drop")
    analysis_df["volatility_tier"] = pd.qcut(analysis_df["demand_volatility"].fillna(0), q=3, labels=["Low Volatility", "Mid Volatility", "High Volatility"], duplicates="drop")
    analysis_df["stockout_hist_tier"] = pd.cut(analysis_df["stockout_freq_7d"], bins=[-0.01, 0.1, 0.4, 1.0], labels=["Low Stockout History", "Moderate History", "Chronic Stockouts"])

    # Slice evaluation summaries
    slices = []
    
    def compute_slice_metrics(grp_col: str, grp_name: str):
        grouped = analysis_df.groupby(grp_col, observed=False)
        for val, group in grouped:
            total = len(group)
            if total == 0:
                continue
            err_cnt = int(group["is_error"].sum())
            fp_cnt = int((group["error_type"] == "False Positive (FP)").sum())
            fn_cnt = int((group["error_type"] == "False Negative (FN)").sum())
            slices.append({
                "Slice Category": grp_name,
                "Subgroup": str(val),
                "Sample Count": total,
                "Error Count": err_cnt,
                "Error Rate (%)": round((err_cnt / total) * 100, 2),
                "False Positive Count": fp_cnt,
                "False Negative Count": fn_cnt,
                "Mean Predicted Prob": round(float(group["predicted_prob"].mean()), 3),
                "Actual Stockout Rate (%)": round(float(group["true_label"].mean() * 100), 2)
            })

    compute_slice_metrics("sales_tier", "Sales Volume Tier")
    compute_slice_metrics("volatility_tier", "Demand Volatility Tier")
    compute_slice_metrics("stockout_hist_tier", "Historical Stockout Frequency")
    compute_slice_metrics("activity_flag", "Promotional Campaign")
    compute_slice_metrics("holiday_flag", "Holiday Period")

    error_summary_df = pd.DataFrame(slices)
    error_summary_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved slice-based error analysis to {output_csv_path}")

    # Generate Error Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error rates by promotional campaign
    promo_slice = error_summary_df[error_summary_df["Slice Category"] == "Promotional Campaign"]
    sns.barplot(data=promo_slice, x="Subgroup", y="Error Rate (%)", ax=axes[0], palette=["#2c7fb8", "#f03b20"])
    axes[0].set_title("Error Rate by Promotional Campaign (0 = Regular, 1 = Promo)")
    axes[0].set_ylabel("Error Rate (%)")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha="center", va="center", color="white", fontweight="bold")

    # Error rates by historical stockout frequency
    hist_slice = error_summary_df[error_summary_df["Slice Category"] == "Historical Stockout Frequency"]
    sns.barplot(data=hist_slice, x="Subgroup", y="Error Rate (%)", ax=axes[1], palette="Blues_r")
    axes[1].set_title("Error Rate by Prior 7-Day Stockout Frequency")
    axes[1].set_ylabel("Error Rate (%)")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_figures_dir, "14_error_rate_by_promo_and_history.png"))
    plt.close()

    # Visualizing Volatility vs Sales Volume Error Heatmap
    pivot_err = analysis_df.pivot_table(index="sales_tier", columns="volatility_tier", values="is_error", aggfunc=lambda x: x.mean() * 100, observed=False)
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot_err, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Error Rate (%)"})
    plt.title("Error Rate (%) across Demand Volume & Volatility Slices")
    plt.xlabel("Demand Volatility")
    plt.ylabel("Sales Volume Tier")
    plt.tight_layout()
    plt.savefig(os.path.join(output_figures_dir, "15_error_rate_volume_volatility_heatmap.png"))
    plt.close()

    # Sample Inspection: Top False Positives & False Negatives
    fp_samples = analysis_df[analysis_df["error_type"] == "False Positive (FP)"].sort_values("predicted_prob", ascending=False).head(5)
    fn_samples = analysis_df[analysis_df["error_type"] == "False Negative (FN)"].sort_values("predicted_prob", ascending=True).head(5)

    sample_cols = ["dt", "store_id", "product_id", "sale_amount", "demand_volatility", "stockout_freq_7d", "activity_flag", "predicted_prob", "true_label"]
    logger.info("Sample False Positives (High Confidence In-Stock Over-prediction):\n" + fp_samples[sample_cols].to_string(index=False))
    logger.info("Sample False Negatives (High Confidence Stockout Under-prediction):\n" + fn_samples[sample_cols].to_string(index=False))

    return error_summary_df

if __name__ == "__main__":
    run_error_analysis()
