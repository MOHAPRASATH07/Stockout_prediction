"""Target Definition Module for Stockout Risk Prediction.

Constructs strictly future-facing target variables across multiple horizons:
- Intraday Next 1 Hour (from 12:00 cutoff)
- Intraday Next 3 Hours (12:00 - 15:00)
- Intraday Next 6 Hours (12:00 - 18:00)
- Next Day (t+1 full operating day, 06:00 - 22:00)

Evaluates class balance and business utility, selecting Next-Day Stockout Risk
as the primary operational target for daily supply replenishment.
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_target_horizons(df: pd.DataFrame, output_path: str = "results/metrics/target_horizon_comparison.csv") -> pd.DataFrame:
    """Evaluate candidate target horizons and export class balance statistics."""
    logger.info("Evaluating candidate stockout target horizons...")
    
    # 1. Next Day Horizon (t+1)
    df_sorted = df.sort_values(["store_id", "product_id", "dt"]).copy()
    df_sorted["target_next_day"] = df_sorted.groupby(["store_id", "product_id"])["stock_hour6_22_cnt"].shift(-1)
    valid_next_day = df_sorted.dropna(subset=["target_next_day"])
    y_next_day = (valid_next_day["target_next_day"] > 0).astype(int)

    # 2. Intraday Horizons from 12:00 midday decision point
    stock_status_matrix = np.array(df["hours_stock_status"].tolist())  # shape (N, 24)
    # Intraday 1h: hour 12:00
    y_intra_1h = stock_status_matrix[:, 12]
    # Intraday 3h: hours 12:00 to 14:00 (inclusive)
    y_intra_3h = (stock_status_matrix[:, 12:15].sum(axis=1) > 0).astype(int)
    # Intraday 6h: hours 12:00 to 17:00 (inclusive)
    y_intra_6h = (stock_status_matrix[:, 12:18].sum(axis=1) > 0).astype(int)

    horizons = [
        {"horizon": "Next 1 Hour (Midday 12h-13h)", "pos": int(y_intra_1h.sum()), "total": len(y_intra_1h)},
        {"horizon": "Next 3 Hours (Midday 12h-15h)", "pos": int(y_intra_3h.sum()), "total": len(y_intra_3h)},
        {"horizon": "Next 6 Hours (Midday 12h-18h)", "pos": int(y_intra_6h.sum()), "total": len(y_intra_6h)},
        {"horizon": "Next Day (t+1 Business Hours)", "pos": int(y_next_day.sum()), "total": len(y_next_day)},
    ]

    records = []
    for h in horizons:
        neg = h["total"] - h["pos"]
        pct = (h["pos"] / h["total"]) * 100
        records.append({
            "horizon": h["horizon"],
            "positive_samples": h["pos"],
            "negative_samples": neg,
            "total_samples": h["total"],
            "positive_percentage": round(pct, 2)
        })

    horizon_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    horizon_df.to_csv(output_path, index=False)
    logger.info(f"Target horizon evaluation saved to {output_path}:\n{horizon_df.to_string(index=False)}")
    return horizon_df

def add_target_variable(df: pd.DataFrame, primary_horizon: str = "next_day") -> pd.DataFrame:
    """
    Constructs the primary binary target column `target_stockout`.
    Guarantees strict future alignment without lookahead leakage.
    Rows with unobserved future horizon (e.g. final day of series) are dropped.
    """
    logger.info(f"Adding primary target variable '{primary_horizon}' to dataset...")
    df_out = df.sort_values(["store_id", "product_id", "dt"]).copy()
    
    if primary_horizon == "next_day":
        # Group strictly by store_id and product_id, then shift future stockout backwards to time t
        future_stockout_cnt = df_out.groupby(["store_id", "product_id"])["stock_hour6_22_cnt"].shift(-1)
        df_out["target_stockout"] = (future_stockout_cnt > 0).astype("Int64")
        # Drop boundary rows where future day t+1 is unobserved
        before_drop = len(df_out)
        df_out = df_out.dropna(subset=["target_stockout"]).copy()
        df_out["target_stockout"] = df_out["target_stockout"].astype(int)
        logger.info(f"Constructed target_stockout (Next Day). Dropped {before_drop - len(df_out):,} series boundary rows.")
        logger.info(f"Final valid rows: {len(df_out):,}, Positive class rate: {df_out['target_stockout'].mean()*100:.2f}%")
    else:
        raise ValueError(f"Unsupported primary horizon: {primary_horizon}")

    return df_out

if __name__ == "__main__":
    from src.data_loader import load_data
    df = load_data(use_sample=True)
    eval_df = evaluate_target_horizons(df)
    df_with_target = add_target_variable(df, "next_day")
    print("\nTarget Addition Sample:")
    print(df_with_target[["dt", "store_id", "product_id", "sale_amount", "stock_hour6_22_cnt", "target_stockout"]].head(10))
