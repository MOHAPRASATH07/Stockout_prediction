"""Master Pipeline Runner for StockGuard: Fresh Retail Replenishment Prioritization.

Executes the entire leakage-free machine learning workflow end-to-end:
1. Data Ingestion & Sampling (src.data_loader)
2. Exploratory Data Analysis & Figures (src.eda)
3. Target Horizon Formulation (src.target)
4. Chronological Splitting & Leakage Audit (src.split)
5. Feature Engineering & Parquet Caching (src.feature_engineering)
6. Baseline Models Training (src.baseline)
7. Advanced XGBoost Training with Early Stopping (src.advanced_model)
8. Bayesian Hyperparameter Optimization (src.optimize)
9. StockGuard Multi-Model Suite, Calibration, Recall@K & Cost Curves (src.models_comparison)
10. Granular Error Analysis across Slices (src.error_analysis)
11. SHAP Global & Local Explainability & Robustness (src.explainability, src.robustness)
12. Academic Word Report Generation (build_word_report)
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StockGuardRunner")

def run_master_pipeline():
    total_start = time.time()
    logger.info("================================================================================")
    logger.info("  STARTING END-TO-END STOCKGUARD MACHINE LEARNING PIPELINE EXECUTION            ")
    logger.info("  Project: StockGuard - Cost-Sensitive Stockout Risk & Replenishment Prioritization")
    logger.info("  Dataset: FreshRetailNet-50K (Dingdong-Inc)                                   ")
    logger.info("================================================================================")

    # Step 1: Data Ingestion
    logger.info("\n--- [Step 1/12] Ingesting Data & Building Data Dictionary ---")
    from src.data_loader import load_data
    df = load_data(use_sample=True)

    # Step 2: Exploratory Data Analysis
    logger.info("\n--- [Step 2/12] Running Exploratory Data Analysis (EDA) ---")
    from src.eda import run_eda
    run_eda()

    # Step 3: Target Formulation
    logger.info("\n--- [Step 3/12] Defining Stockout Horizons & Class Balance ---")
    from src.target import evaluate_target_horizons, add_target_variable
    evaluate_target_horizons(df)
    df = add_target_variable(df, "next_day")

    # Step 4: Temporal Splitting & Leakage Verification
    logger.info("\n--- [Step 4/12] Performing Chronological Train/Val/Test Split ---")
    from src.split import temporal_split
    train_df, val_df, test_df, _ = temporal_split(df)

    # Step 5: Feature Engineering
    logger.info("\n--- [Step 5/12] Executing Leakage-Free Feature Engineering ---")
    from src.feature_engineering import prepare_feature_pipeline, assert_feature_leakage_isolation
    prepare_feature_pipeline(train_df, val_df, test_df)
    assert_feature_leakage_isolation(df)

    # Step 6: Baseline Models
    logger.info("\n--- [Step 6/12] Training Baseline Models (Dummy, Logistic, Tree) ---")
    from src.baseline import train_baselines
    train_baselines()

    # Step 7: Advanced XGBoost Model
    logger.info("\n--- [Step 7/12] Training Advanced XGBoost Model with Early Stopping ---")
    from src.advanced_model import train_xgboost
    train_xgboost()

    # Step 8: Optuna Hyperparameter Optimization
    logger.info("\n--- [Step 8/12] Running Optuna Bayesian Hyperparameter Optimization ---")
    from src.optimize import run_hyperparameter_optimization
    run_hyperparameter_optimization(n_trials=10)

    # Step 9: StockGuard Multi-Model Comparison, Probability Calibration & Recall@K
    logger.info("\n--- [Step 9/12] Training Multi-Model Suite, Calibration & Recall@K Capacity ---")
    from src.models_comparison import train_and_benchmark_all
    train_and_benchmark_all()

    # Step 10: Error Analysis & Slices
    logger.info("\n--- [Step 10/12] Performing Slice-Based Error Analysis ---")
    from src.error_analysis import run_error_analysis
    run_error_analysis()

    # Step 11: SHAP Explainability & Robustness
    logger.info("\n--- [Step 11/12] Computing SHAP Values & Operational Robustness ---")
    from src.explainability import run_shap_explainability
    from src.robustness import run_robustness_experiments
    run_shap_explainability(sample_size=1000)
    run_robustness_experiments()

    # Step 12: Build Final Academic Word Report
    logger.info("\n--- [Step 12/12] Building Academic Microsoft Word (.docx) Paper ---")
    from build_word_report import generate_docx_report
    generate_docx_report("Moha_Prasath_AML_Project_Report.docx")

    total_elapsed = time.time() - total_start
    logger.info("================================================================================")
    logger.info(f"  STOCKGUARD MASTER PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.1f} SECONDS! ")
    logger.info("  All figures saved to: results/figures/                                        ")
    logger.info("  All metrics saved to: results/metrics/                                        ")
    logger.info("  All models saved to:  models/                                                 ")
    logger.info("  Reports generated:    Moha_Prasath_AML_Project_Report.docx                    ")
    logger.info("                        StockGuard_AML_Project_Report.docx                      ")
    logger.info("================================================================================")

if __name__ == "__main__":
    run_master_pipeline()
