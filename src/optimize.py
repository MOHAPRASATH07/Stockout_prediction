"""Bayesian Hyperparameter Optimization using Optuna for XGBoost.

Optimizes Validation PR-AUC strictly on Training/Validation splits.
Test data remains held-out and completely untouched during optimization.
"""

import os
import sys
import json
import time
import logging
import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.target import add_target_variable
from src.split import temporal_split
from src.feature_engineering import prepare_feature_pipeline

# Configure logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_hyperparameter_optimization(
    n_trials: int = 15,
    output_trials_csv: str = "results/metrics/optuna_trials.csv",
    output_best_params_json: str = "models/best_parameters.json",
    output_best_model_path: str = "models/best_model.pkl"
) -> dict:
    """Run Optuna Bayesian optimization maximizing Validation PR-AUC."""
    os.makedirs(os.path.dirname(output_trials_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_best_params_json), exist_ok=True)

    # 1. Load precomputed features
    df = load_data(use_sample=True)
    df = add_target_variable(df, "next_day")
    train_df, val_df, test_df, _ = temporal_split(df)
    train_f, val_f, test_f, feature_cols = prepare_feature_pipeline(train_df, val_df, test_df)

    X_train, y_train = train_f[feature_cols].fillna(0), train_f["target_stockout"]
    X_val, y_val = val_f[feature_cols].fillna(0), val_f["target_stockout"]

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    start_time = time.time()
    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 250,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "scale_pos_weight": float(scale_pos_weight),
            "random_state": 42,
            "eval_metric": "aucpr",
            "early_stopping_rounds": 25,
            "n_jobs": -1
        }

        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_probs = clf.predict_proba(X_val)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, val_probs))

        trial_records.append({
            "trial_number": trial.number,
            "val_pr_auc": round(val_pr_auc, 5),
            "learning_rate": round(params["learning_rate"], 4),
            "max_depth": params["max_depth"],
            "min_child_weight": params["min_child_weight"],
            "subsample": round(params["subsample"], 3),
            "colsample_bytree": round(params["colsample_bytree"], 3),
            "gamma": round(params["gamma"], 3),
            "best_iteration": int(clf.best_iteration)
        })

        return val_pr_auc

    logger.info(f"Starting Optuna search ({n_trials} trials) optimizing Validation PR-AUC...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    elapsed_time = time.time() - start_time
    logger.info(f"Optimization finished in {elapsed_time:.1f} seconds.")

    # Save trial history
    trials_df = pd.DataFrame(trial_records).sort_values("val_pr_auc", ascending=False)
    trials_df.to_csv(output_trials_csv, index=False)

    best_params = study.best_params
    best_pr_auc = float(study.best_value)

    summary = {
        "best_parameters": best_params,
        "best_validation_pr_auc": round(best_pr_auc, 5),
        "number_of_trials": n_trials,
        "total_optimization_time_seconds": round(elapsed_time, 2)
    }

    with open(output_best_params_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Best Validation PR-AUC: {best_pr_auc:.5f}")
    logger.info(f"Best Hyperparameters: {best_params}")

    # Retrain best model configuration with full early stopping
    full_best_params = {
        "n_estimators": 400,
        "learning_rate": best_params["learning_rate"],
        "max_depth": best_params["max_depth"],
        "min_child_weight": best_params["min_child_weight"],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "gamma": best_params["gamma"],
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": 42,
        "eval_metric": ["logloss", "aucpr"],
        "early_stopping_rounds": 30,
        "n_jobs": -1
    }

    best_clf = xgb.XGBClassifier(**full_best_params)
    best_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(best_clf, output_best_model_path)
    logger.info(f"Saved tuned best model to {output_best_model_path}")

    # Save summary report
    summary_txt = f"""=======================================================
OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION SUMMARY
=======================================================
Trials Evaluated: {n_trials}
Search Space: learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma
Objective Metric: Validation PR-AUC (chronological out-of-fold)
Best Validation PR-AUC: {best_pr_auc:.5f}
Total Search Runtime: {elapsed_time:.1f}s

Optimal Hyperparameters:
{json.dumps(best_params, indent=2)}
=======================================================
"""
    with open("results/metrics/optuna_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)

    return summary

if __name__ == "__main__":
    summary = run_hyperparameter_optimization(n_trials=10)
    print(json.dumps(summary, indent=2))
