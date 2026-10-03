"""SHAP Explainability Module for Stockout Risk Prediction.

Computes game-theoretic Shapley values using TreeExplainer for the tuned XGBoost model:
- Global feature importance (Mean |SHAP|)
- Beeswarm summary plots (feature directionality & non-linear thresholds)
- Dependence plots for critical demand & inventory drivers
- Local waterfall explanations for True Positives, False Positives, and False Negatives
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.target import add_target_variable
from src.split import temporal_split
from src.feature_engineering import prepare_feature_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight"})

def run_shap_explainability(
    model_path: str = "models/best_model.pkl",
    fallback_path: str = "models/xgboost_model.joblib",
    output_dir: str = "results/figures/shap",
    metrics_path: str = "results/metrics/top_features.csv",
    sample_size: int = 1500
) -> pd.DataFrame:
    """Compute and save SHAP explanations."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # 1. Load data and features
    df = load_data(use_sample=True)
    df = add_target_variable(df, "next_day")
    train_df, val_df, test_df, _ = temporal_split(df)
    train_f, val_f, test_f, feature_cols = prepare_feature_pipeline(train_df, val_df, test_df)

    active_path = model_path if os.path.exists(model_path) else fallback_path
    logger.info(f"Loading trained XGBoost model from {active_path}...")
    model = joblib.load(active_path)

    X_test = test_f[feature_cols].fillna(0)
    y_test = test_f["target_stockout"].values

    # Sample a representative subset for SHAP computation
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_indices].copy()
    y_sample = y_test[sample_indices]

    logger.info(f"Computing TreeExplainer SHAP values on {len(X_sample)} test samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # 2. Global Feature Importance Ranking (Mean |SHAP|)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    
    top_df["rank"] = top_df.index + 1
    top_df.to_csv(metrics_path, index=False)
    logger.info(f"Top 10 Most Influential Features:\n{top_df.head(10).to_string(index=False)}")

    # 3. SHAP Bar Summary Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar_importance.png"))
    plt.close()

    # 4. SHAP Beeswarm Plot (Directional Impacts)
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Beeswarm Summary Plot: Directional Feature Impact on Stockout Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"))
    plt.close()

    # 5. SHAP Dependence Plots for Top Drivers
    top_features = top_df["feature"].head(3).tolist()
    for feat in top_features:
        plt.figure(figsize=(8, 5))
        shap.plots.scatter(shap_values[:, feat], color=shap_values, show=False)
        plt.title(f"SHAP Dependence Plot for '{feat}'")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_dependence_{feat}.png"))
        plt.close()

    # 6. Local Waterfall Plots for Specific Diagnostic Cases
    probs = model.predict_proba(X_sample)[:, 1]
    preds = (probs >= 0.5).astype(int)

    tp_idx = np.where((y_sample == 1) & (preds == 1))[0]
    fp_idx = np.where((y_sample == 0) & (preds == 1))[0]
    fn_idx = np.where((y_sample == 1) & (preds == 0))[0]

    # Save representative waterfalls
    cases = [("true_positive", tp_idx), ("false_positive", fp_idx), ("false_negative", fn_idx)]
    for case_name, idx_arr in cases:
        if len(idx_arr) > 0:
            target_idx = int(idx_arr[0])
            plt.figure(figsize=(9, 5))
            shap.plots.waterfall(shap_values[target_idx], max_display=10, show=False)
            plt.title(f"Local SHAP Explanation: {case_name.replace('_', ' ').title()}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_waterfall_{case_name}.png"))
            plt.close()

    logger.info(f"SHAP explainability plots successfully saved to {output_dir}")
    return top_df

if __name__ == "__main__":
    run_shap_explainability()
