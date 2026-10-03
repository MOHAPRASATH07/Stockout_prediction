"""Temporal Split Module for Fresh Retail Time Series.

Implements strict, leakage-free chronological splitting into Train, Validation, and Test sets.
Validates that future records never contaminate earlier splits.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def temporal_split(
    df: pd.DataFrame,
    date_col: str = "dt",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_summary_path: str = "results/metrics/split_summary.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs strict chronological splitting across all series.
    Guarantees:
      max(Train_date) < min(Val_date) <= max(Val_date) < min(Test_date)
    """
    logger.info("Performing chronological temporal split...")
    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
    
    unique_dates = sorted(df_sorted[date_col].unique())
    n_dates = len(unique_dates)
    
    train_idx = int(np.floor(n_dates * train_ratio))
    val_idx = int(np.floor(n_dates * (train_ratio + val_ratio)))

    train_dates = unique_dates[:train_idx]
    val_dates = unique_dates[train_idx:val_idx]
    test_dates = unique_dates[val_idx:]

    train_df = df_sorted[df_sorted[date_col].isin(train_dates)].copy()
    val_df = df_sorted[df_sorted[date_col].isin(val_dates)].copy()
    test_df = df_sorted[df_sorted[date_col].isin(test_dates)].copy()

    # Determine target column if present
    target_col = "target_stockout" if "target_stockout" in df_sorted.columns else "is_stockout_day"
    if target_col not in df_sorted.columns and "stock_hour6_22_cnt" in df_sorted.columns:
        train_df["temp_target"] = (train_df["stock_hour6_22_cnt"] > 0).astype(int)
        val_df["temp_target"] = (val_df["stock_hour6_22_cnt"] > 0).astype(int)
        test_df["temp_target"] = (test_df["stock_hour6_22_cnt"] > 0).astype(int)
        target_col = "temp_target"

    summary_records = [
        {
            "split": "Train",
            "start_date": train_dates[0].strftime("%Y-%m-%d"),
            "end_date": train_dates[-1].strftime("%Y-%m-%d"),
            "num_days": len(train_dates),
            "num_rows": len(train_df),
            "positive_count": int(train_df[target_col].sum()) if target_col in train_df.columns else None,
            "positive_rate_pct": round(train_df[target_col].mean() * 100, 2) if target_col in train_df.columns else None
        },
        {
            "split": "Validation",
            "start_date": val_dates[0].strftime("%Y-%m-%d"),
            "end_date": val_dates[-1].strftime("%Y-%m-%d"),
            "num_days": len(val_dates),
            "num_rows": len(val_df),
            "positive_count": int(val_df[target_col].sum()) if target_col in val_df.columns else None,
            "positive_rate_pct": round(val_df[target_col].mean() * 100, 2) if target_col in val_df.columns else None
        },
        {
            "split": "Test",
            "start_date": test_dates[0].strftime("%Y-%m-%d"),
            "end_date": test_dates[-1].strftime("%Y-%m-%d"),
            "num_days": len(test_dates),
            "num_rows": len(test_df),
            "positive_count": int(test_df[target_col].sum()) if target_col in test_df.columns else None,
            "positive_rate_pct": round(test_df[target_col].mean() * 100, 2) if target_col in test_df.columns else None
        }
    ]

    summary_df = pd.DataFrame(summary_records)
    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    summary_df.to_csv(output_summary_path, index=False)
    logger.info(f"Split Summary:\n{summary_df.to_string(index=False)}")

    # Verification assertions
    verify_no_temporal_leakage(train_df, val_df, test_df, date_col)
    return train_df, val_df, test_df, summary_df

def verify_no_temporal_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "dt"
) -> bool:
    """Automated assertion checks verifying strict temporal isolation."""
    train_max = train_df[date_col].max()
    val_min = val_df[date_col].min()
    val_max = val_df[date_col].max()
    test_min = test_df[date_col].min()

    assert train_max < val_min, f"Temporal Leakage! Train max date ({train_max}) >= Val min date ({val_min})"
    assert val_max < test_min, f"Temporal Leakage! Val max date ({val_max}) >= Test min date ({test_min})"

    # Disjoint date sets
    train_set = set(train_df[date_col].unique())
    val_set = set(val_df[date_col].unique())
    test_set = set(test_df[date_col].unique())

    assert train_set.isdisjoint(val_set), "Train and Val dates overlap!"
    assert val_set.isdisjoint(test_set), "Val and Test dates overlap!"
    assert train_set.isdisjoint(test_set), "Train and Test dates overlap!"

    logger.info("Verification PASSED: Zero temporal overlap or lookahead leakage detected.")
    return True

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.target import add_target_variable

    df = load_data(use_sample=True)
    df = add_target_variable(df, "next_day")
    train_df, val_df, test_df, summary_df = temporal_split(df)
    print("Temporal splitting module validated successfully.")
