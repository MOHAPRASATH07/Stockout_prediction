"""StockGuard Multi-Model Training, Operational Benchmarking, and Probability Calibration Module.

Trains and evaluates:
1. Baseline: Dummy Classifier (Stratified)
2. Baseline: Logistic Regression (StandardScaler + L2)
3. Benchmark: Decision Tree Classifier
4. Nonlinear Ensemble: Random Forest Classifier
5. Gradient Boosting: XGBoost (Optuna Tuned)
6. Gradient Boosting: LightGBM (Leaf-wise with early stopping)
7. Gradient Boosting: CatBoost (Symmetric trees with early stopping)
8. Champion Calibrated: Isotonic Probability Calibration on Validation split

Evaluates:
- Statistical Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score
- Operational Capacity Metrics: Recall@1%, Recall@2%, Recall@5%, Recall@10%
- Operational Precision Metrics: Precision@1%, Precision@2%, Precision@5%, Precision@10%
- Cost-Benefit Tradeoff Curve: Optimal replenishment inspection threshold under asymmetric costs.
"""

import os
import sys
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    brier_score_loss, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight"})

class CalibratedChampion:
    """Wrapper that applies an Isotonic Regression calibrator on top of a fitted classifier."""
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        if hasattr(self.base_model, "predict_proba"):
            raw_probs = self.base_model.predict_proba(X)[:, 1]
        else:
            raw_probs = self.base_model.predict(X).astype(float)
        cal_probs = np.clip(self.calibrator.predict(raw_probs), 0.0, 1.0)
        return np.vstack([1.0 - cal_probs, cal_probs]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

CalibratedChampion.__module__ = "src.models_comparison"

def compute_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_pct: float) -> float:
    """Calculate Recall@K%: fraction of all true stockouts captured in the top K% highest-risk predictions."""
    total_positives = np.sum(y_true == 1)
    if total_positives == 0:
        return 0.0
    n = len(y_true)
    top_m = max(1, int(np.ceil((k_pct / 100.0) * n)))
    sorted_indices = np.argsort(y_prob)[::-1]
    top_indices = sorted_indices[:top_m]
    caught_positives = np.sum(y_true[top_indices] == 1)
    return float(caught_positives / total_positives)

def compute_precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_pct: float) -> float:
    """Calculate Precision@K%: hit rate within the top K% highest-risk predictions."""
    n = len(y_true)
    top_m = max(1, int(np.ceil((k_pct / 100.0) * n)))
    sorted_indices = np.argsort(y_prob)[::-1]
    top_indices = sorted_indices[:top_m]
    caught_positives = np.sum(y_true[top_indices] == 1)
    return float(caught_positives / top_m)

def train_and_benchmark_all():
    """Run full StockGuard model training, calibration, and operational evaluation."""
    os.makedirs("models/baseline", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)

    # 1. Load Parquet Features
    logger.info("Loading cached feature parquets (leakage-free temporal splits)...")
    train_f = pd.read_parquet("data/processed/train_features.parquet")
    val_f = pd.read_parquet("data/processed/val_features.parquet")
    test_f = pd.read_parquet("data/processed/test_features.parquet")

    feature_cols = get_feature_columns()
    logger.info(f"Feature set size: {len(feature_cols)} features.")

    X_train = train_f[feature_cols].fillna(0)
    y_train = train_f["target_stockout"].values

    X_val = val_f[feature_cols].fillna(0)
    y_val = val_f["target_stockout"].values

    X_test = test_f[feature_cols].fillna(0)
    y_test = test_f["target_stockout"].values

    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos = neg_count / max(pos_count, 1)
    logger.info(f"Train split: {len(X_train):,} rows. Target ratio pos/neg: {pos_count:,} / {neg_count:,} (scale_pos_weight: {scale_pos:.3f})")

    models = {}

    # --- 1. Dummy Classifier ---
    dummy_path = "models/baseline/dummy_(stratified).joblib"
    if os.path.exists(dummy_path):
        models["Dummy (Stratified)"] = joblib.load(dummy_path)
    else:
        logger.info("Fitting Dummy Classifier...")
        dummy = DummyClassifier(strategy="stratified", random_state=42)
        dummy.fit(X_train, y_train)
        joblib.dump(dummy, dummy_path)
        models["Dummy (Stratified)"] = dummy

    # --- 2. Logistic Regression ---
    lr_path = "models/baseline/logistic_regression.joblib"
    if os.path.exists(lr_path):
        models["Logistic Regression"] = joblib.load(lr_path)
    else:
        logger.info("Fitting Logistic Regression Pipeline...")
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])
        lr.fit(X_train, y_train)
        joblib.dump(lr, lr_path)
        models["Logistic Regression"] = lr

    # --- 3. Decision Tree ---
    dt_path = "models/baseline/decision_tree.joblib"
    if os.path.exists(dt_path):
        models["Decision Tree"] = joblib.load(dt_path)
    else:
        logger.info("Fitting Decision Tree Benchmark...")
        dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
        dt.fit(X_train, y_train)
        joblib.dump(dt, dt_path)
        models["Decision Tree"] = dt

    # --- 4. Random Forest ---
    rf_path = "models/baseline/random_forest.joblib"
    if os.path.exists(rf_path):
        logger.info("Loading cached Random Forest...")
        models["Random Forest"] = joblib.load(rf_path)
    else:
        logger.info("Training Random Forest Classifier (150 trees, max_depth=12)...")
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        joblib.dump(rf, rf_path)
        models["Random Forest"] = rf
        logger.info(f"Saved Random Forest to {rf_path}")

    # --- 5. XGBoost (Tuned) ---
    xgb_best_path = "models/best_model.pkl" if os.path.exists("models/best_model.pkl") else "models/xgboost_model.joblib"
    if os.path.exists(xgb_best_path):
        logger.info(f"Loading tuned XGBoost from {xgb_best_path}...")
        models["XGBoost (Tuned)"] = joblib.load(xgb_best_path)
    else:
        logger.info("Training XGBoost Classifier...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=float(scale_pos),
            random_state=42,
            eval_metric=["logloss", "aucpr"],
            early_stopping_rounds=30,
            n_jobs=-1
        )
        xgb_clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        joblib.dump(xgb_clf, xgb_best_path)
        models["XGBoost (Tuned)"] = xgb_clf

    # --- 6. LightGBM ---
    lgb_path = "models/lightgbm_model.joblib"
    if os.path.exists(lgb_path):
        logger.info("Loading cached LightGBM...")
        models["LightGBM"] = joblib.load(lgb_path)
    else:
        logger.info("Training LightGBM Classifier with early stopping...")
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=35,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_samples=20,
            scale_pos_weight=float(scale_pos),
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        lgb_clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        joblib.dump(lgb_clf, lgb_path)
        models["LightGBM"] = lgb_clf
        logger.info(f"Saved LightGBM to {lgb_path}")

    # --- 7. CatBoost ---
    cb_path = "models/catboost_model.joblib"
    if os.path.exists(cb_path):
        logger.info("Loading cached CatBoost...")
        models["CatBoost"] = joblib.load(cb_path)
    else:
        logger.info("Training CatBoost Classifier with early stopping...")
        cb_clf = cb.CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            eval_metric="PRAUC",
            early_stopping_rounds=30,
            random_seed=42,
            verbose=False,
            thread_count=-1
        )
        cb_clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        joblib.dump(cb_clf, cb_path)
        models["CatBoost"] = cb_clf
        logger.info(f"Saved CatBoost to {cb_path}")

    # Identify Champion on Validation PR-AUC
    val_pr_scores = {}
    for name, m in models.items():
        if hasattr(m, "predict_proba"):
            p_val = m.predict_proba(X_val)[:, 1]
        else:
            p_val = m.predict(X_val).astype(float)
        val_pr_scores[name] = average_precision_score(y_val, p_val)

    champ_name = max(val_pr_scores, key=val_pr_scores.get)
    logger.info(f"Validation PR-AUC Champion: {champ_name} (PR-AUC = {val_pr_scores[champ_name]:.4f})")

    # --- 8. Probability Calibration (Champion Model) ---
    calib_path = "models/champion_calibrated.joblib"
    logger.info(f"Calibrating {champ_name} using Isotonic Regression on held-out Validation set (leakage-free)...")
    base_champ = models[champ_name]
    val_raw_probs = base_champ.predict_proba(X_val)[:, 1] if hasattr(base_champ, "predict_proba") else base_champ.predict(X_val).astype(float)

    iso_calibrator = IsotonicRegression(out_of_bounds="clip")
    iso_calibrator.fit(val_raw_probs, y_val)

    calibrated_clf = CalibratedChampion(base_champ, iso_calibrator)
    joblib.dump(calibrated_clf, calib_path)
    champ_cal_key = f"{champ_name} (Calibrated)"
    models[champ_cal_key] = calibrated_clf
    logger.info(f"Saved Calibrated Champion to {calib_path}")

    # 2. Comprehensive Test Set Evaluation
    logger.info("Evaluating all candidate models on held-out Test set (28,000 rows)...")
    metrics_records = []
    recall_k_records = []
    model_test_probs = {}

    colors = {
        "Dummy (Stratified)": "#8c8c8c",
        "Logistic Regression": "#3b82f6",
        "Decision Tree": "#8b5cf6",
        "Random Forest": "#10b981",
        "XGBoost (Tuned)": "#f97316",
        "LightGBM": "#06b6d4",
        "CatBoost": "#ec4899",
        champ_cal_key: "#b91c1c"
    }

    for name, m in models.items():
        if hasattr(m, "predict_proba"):
            probs = m.predict_proba(X_test)[:, 1]
        else:
            probs = m.predict(X_test).astype(float)
        
        preds = (probs >= 0.5).astype(int)
        model_test_probs[name] = probs

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        try:
            roc = roc_auc_score(y_test, probs)
        except Exception:
            roc = 0.5
        try:
            pr = average_precision_score(y_test, probs)
        except Exception:
            pr = float(np.mean(y_test))
        brier = brier_score_loss(y_test, probs)

        # Operational metrics at K = 1%, 2%, 5%, 10%
        rec_1 = compute_recall_at_k(y_test, probs, 1.0)
        rec_2 = compute_recall_at_k(y_test, probs, 2.0)
        rec_5 = compute_recall_at_k(y_test, probs, 5.0)
        rec_10 = compute_recall_at_k(y_test, probs, 10.0)

        prec_1 = compute_precision_at_k(y_test, probs, 1.0)
        prec_2 = compute_precision_at_k(y_test, probs, 2.0)
        prec_5 = compute_precision_at_k(y_test, probs, 5.0)
        prec_10 = compute_precision_at_k(y_test, probs, 10.0)

        metrics_records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4),
            "PR-AUC (Primary)": round(pr, 4),
            "Brier Score": round(brier, 4),
            "Recall@1%": round(rec_1, 4),
            "Recall@5%": round(rec_5, 4),
            "Recall@10%": round(rec_10, 4)
        })

        recall_k_records.append({
            "Model": name,
            "Recall@1%": round(rec_1, 4),
            "Precision@1%": round(prec_1, 4),
            "Recall@2%": round(rec_2, 4),
            "Precision@2%": round(prec_2, 4),
            "Recall@5%": round(rec_5, 4),
            "Precision@5%": round(prec_5, 4),
            "Recall@10%": round(rec_10, 4),
            "Precision@10%": round(prec_10, 4)
        })

    # Save Metrics DataFrames
    comp_df = pd.DataFrame(metrics_records).sort_values("PR-AUC (Primary)", ascending=False)
    comp_csv = "results/metrics/stockguard_model_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    comp_df.to_csv("results/metrics/final_model_comparison.csv", index=False)
    logger.info(f"StockGuard Multi-Model Evaluation Table:\n{comp_df.to_string(index=False)}")

    rec_k_df = pd.DataFrame(recall_k_records)
    rec_k_csv = "results/metrics/operational_recall_at_k.csv"
    rec_k_df.to_csv(rec_k_csv, index=False)

    # 3. Save Test Predictions for Champion
    best_probs = model_test_probs[champ_cal_key]
    pred_export = test_f[["city_id", "store_id", "product_id", "dt", "target_stockout", "sale_amount", "discount"]].copy()
    pred_export["stockout_prob"] = best_probs
    pred_export["calibrated_risk_tier"] = pd.cut(
        best_probs,
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Low (<25%)", "Moderate (25-50%)", "High (50-75%)", "Critical (>75%)"]
    )
    pred_export.to_parquet("results/predictions/stockguard_champion_predictions.parquet", index=False)
    logger.info("Saved champion test predictions to results/predictions/stockguard_champion_predictions.parquet")

    # 4. Publication Plot: ROC Curves
    plt.figure(figsize=(9, 6.5))
    for name, probs in model_test_probs.items():
        if name == "Dummy (Stratified)":
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})", color=colors.get(name, "black"), linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.500)")
    plt.title("StockGuard: Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig("results/figures/11_roc_curves.png")
    plt.close()

    # 5. Publication Plot: Precision-Recall Curves
    baseline_pr = np.mean(y_test)
    plt.figure(figsize=(9, 6.5))
    for name, probs in model_test_probs.items():
        if name == "Dummy (Stratified)":
            continue
        precision, recall, _ = precision_recall_curve(y_test, probs)
        score = average_precision_score(y_test, probs)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {score:.3f})", color=colors.get(name, "black"), linewidth=2)
    plt.axhline(baseline_pr, color="black", linestyle="--", alpha=0.6, label=f"No-Skill Baseline ({baseline_pr:.3f})")
    plt.title("StockGuard: Precision-Recall (PR) Curves (Primary Imbalance Metric)", fontsize=13, fontweight="bold")
    plt.xlabel("Recall (Coverage of True Stockouts)", fontsize=11)
    plt.ylabel("Precision (Replenishment Action Accuracy)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("results/figures/12_pr_curves.png")
    plt.close()

    # 6. Publication Plot: Reliability / Probability Calibration Curves
    plt.figure(figsize=(9, 6.5))
    for name, probs in model_test_probs.items():
        if name == "Dummy (Stratified)":
            continue
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=f"{name}", color=colors.get(name, "black"), linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)
    plt.title("Probability Reliability Curves (Before vs After Isotonic Calibration)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Stockout Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Stockouts", fontsize=11)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("results/figures/13_calibration_curves.png")
    plt.close()

    # 7. Publication Plot: Operational Human-Attention Recall@K Curve
    k_range = np.linspace(0.5, 20, 40)
    plt.figure(figsize=(9, 6.5))
    for name in ["Logistic Regression", "Random Forest", "XGBoost (Tuned)", "LightGBM", "CatBoost", champ_cal_key]:
        if name in model_test_probs:
            rec_curve = [compute_recall_at_k(y_test, model_test_probs[name], k) * 100 for k in k_range]
            plt.plot(k_range, rec_curve, label=name, color=colors.get(name, "black"), linewidth=2.2)
    plt.axvline(1.0, color="gray", linestyle=":", alpha=0.7, label="1% Store Attention")
    plt.axvline(5.0, color="gray", linestyle="--", alpha=0.7, label="5% Store Attention")
    plt.title("StockGuard Operational Capacity Curve: Recall@K% (Top-K Review)", fontsize=13, fontweight="bold")
    plt.xlabel("Review Capacity K% of Total Store-Product SKUs", fontsize=11)
    plt.ylabel("% of All Tomorrow's Stockouts Prevented (Recall@K %)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig("results/figures/14_recall_at_k_curves.png")
    plt.close()

    # 8. Asymmetric Cost Curve Analysis
    # C_FN = $20 (lost margin, perishability disruption), C_FP = $3 (manual staff shelf check), C_TP = $3
    C_FN = 20.0
    C_FP = 3.0
    C_TP = 3.0
    C_TN = 0.0

    thresholds = np.linspace(0.05, 0.95, 91)
    cost_data = []

    probs_champ = model_test_probs[champ_cal_key]
    for th in thresholds:
        pred_th = (probs_champ >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_th).ravel()
        total_cost = (fn * C_FN) + (fp * C_FP) + (tp * C_TP) + (tn * C_TN)
        cost_data.append({
            "threshold": round(th, 3),
            "total_cost": round(total_cost, 2),
            "fn": int(fn),
            "fp": int(fp),
            "tp": int(tp),
            "tn": int(tn),
            "cost_per_sku": round(total_cost / len(y_test), 3)
        })

    cost_df = pd.DataFrame(cost_data)
    cost_df.to_csv("results/metrics/cost_curve_analysis.csv", index=False)

    min_cost_row = cost_df.loc[cost_df["total_cost"].idxmin()]
    opt_th = min_cost_row["threshold"]
    min_cost = min_cost_row["total_cost"]

    # Baseline cost: no intervention (FN = total actual stockouts, cost = pos * C_FN)
    baseline_cost = np.sum(y_test == 1) * C_FN
    cost_savings = baseline_cost - min_cost

    plt.figure(figsize=(9, 6))
    plt.plot(cost_df["threshold"], cost_df["total_cost"] / 1000.0, color="#b91c1c", linewidth=2.5, label="StockGuard Total Cost ($k)")
    plt.axhline(baseline_cost / 1000.0, color="black", linestyle="--", label=f"No-Action Baseline (${baseline_cost/1000:.1f}k)")
    plt.scatter([opt_th], [min_cost / 1000.0], color="#d97706", s=120, zorder=5, label=f"Optimal Threshold = {opt_th:.2f} (${min_cost/1000:.1f}k)")
    plt.title(f"Operational Cost-Utility Tradeoff: Optimal Threshold $\\theta^* = {opt_th:.2f}$", fontsize=13, fontweight="bold")
    plt.xlabel("Replenishment Decision Threshold $\\theta$", fontsize=11)
    plt.ylabel("Total Fleet Operating Cost ($ thousands)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("results/figures/15_cost_tradeoff_curve.png")
    plt.close()

    logger.info(f"Cost-Utility Analysis: Optimal Threshold = {opt_th:.2f}, Total Cost = ${min_cost:,.2f} vs Baseline ${baseline_cost:,.2f} (Savings: ${cost_savings:,.2f} or {cost_savings/baseline_cost*100:.1f}%)")
    logger.info("StockGuard Multi-Model Training and Evaluation Pipeline Completed Successfully!")
    return comp_df, rec_k_df

if __name__ == "__main__":
    train_and_benchmark_all()
