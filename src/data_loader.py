"""Data Loader Module for FreshRetailNet-50K.

Handles downloading, caching, inspecting schema, extracting data dictionaries,
and generating representative developmental samples without modifying raw files.
"""

import os
import sys
import logging
import urllib.request
from typing import Tuple, Dict, Any, Optional
import yaml
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def download_file(url: str, dest_path: str) -> None:
    """Download a file with progress reporting if not already downloaded."""
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"File already exists: {dest_path} ({size_mb:.2f} MB)")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading from {url} to {dest_path} ...")
    
    # User-agent header to avoid 403 from HuggingFace
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FreshRetailNet-Loader/1.0"}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1MB
        downloaded = 0
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                logger.info(f"Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
    logger.info(f"Finished downloading {dest_path}")

def inspect_dataset(df: pd.DataFrame, dataset_name: str = "FreshRetailNet") -> Dict[str, Any]:
    """Inspect dataset statistics, unique entities, missing values, and columns."""
    logger.info(f"=== Dataset Inspection: {dataset_name} ===")
    n_rows, n_cols = df.shape
    logger.info(f"Rows: {n_rows:,}, Columns: {n_cols}")

    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        null_pct = (null_count / n_rows) * 100
        col_info.append({"column": col, "dtype": dtype, "null_count": null_count, "null_pct": null_pct})

    # Summary statistics
    n_stores = df["store_id"].nunique() if "store_id" in df.columns else None
    n_products = df["product_id"].nunique() if "product_id" in df.columns else None
    n_cities = df["city_id"].nunique() if "city_id" in df.columns else None
    
    dt_min = df["dt"].min() if "dt" in df.columns else None
    dt_max = df["dt"].max() if "dt" in df.columns else None

    logger.info(f"Unique Stores: {n_stores:,}")
    logger.info(f"Unique Products: {n_products:,}")
    logger.info(f"Unique Cities: {n_cities}")
    logger.info(f"Date Range: {dt_min} to {dt_max}")

    # Feature categorization check
    detection = {
        "sales": any("sale" in c for c in df.columns),
        "stockout": any("stock" in c for c in df.columns),
        "inventory_status": "hours_stock_status" in df.columns,
        "promotion": any("activ" in c or "promo" in c for c in df.columns),
        "discount": "discount" in df.columns,
        "weather": any(c in df.columns for c in ["precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]),
        "timestamp_date": "dt" in df.columns
    }
    logger.info(f"Detected Modalities: {detection}")

    return {
        "rows": n_rows,
        "cols": n_cols,
        "stores": n_stores,
        "products": n_products,
        "cities": n_cities,
        "date_min": str(dt_min),
        "date_max": str(dt_max),
        "column_info": col_info,
        "detection": detection
    }

def create_data_dictionary(df: pd.DataFrame, output_path: str = "results/metrics/data_dictionary.csv") -> pd.DataFrame:
    """Create and save a comprehensive data dictionary."""
    descriptions = {
        "city_id": "Anonymized identifier for the retail market city",
        "store_id": "Anonymized identifier for the fulfillment store/warehouse",
        "management_group_id": "Operational regional management division ID",
        "first_category_id": "Broad product category (e.g. Vegetables, Meat, Fruit)",
        "second_category_id": "Sub-category identifier",
        "third_category_id": "Granular commodity classification",
        "product_id": "Unique stock-keeping unit (SKU) identifier",
        "dt": "Calendar date of observation (YYYY-MM-DD)",
        "sale_amount": "Total daily recorded sales volume/amount for this SKU-store",
        "hours_sale": "24-element float list of hourly sales distributions (0:00 to 23:00)",
        "stock_hour6_22_cnt": "Total hours out of stock during active operating hours (06:00 to 22:00)",
        "hours_stock_status": "24-element binary vector (1 = out of stock, 0 = in stock) per hour",
        "discount": "Applied promotional discount multiplier (e.g. 1.0 = full price, 0.8 = 20% off)",
        "holiday_flag": "Binary indicator for national statutory holidays",
        "activity_flag": "Binary indicator for marketing campaign/promotional event",
        "precpt": "Recorded precipitation level in mm (meteorological context)",
        "avg_temperature": "Average ambient daily temperature in degrees Celsius",
        "avg_humidity": "Average daily relative humidity percentage",
        "avg_wind_level": "Average Beaufort wind scale level"
    }

    records = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        sample_val = str(df[col].iloc[0]) if len(df) > 0 else ""
        if len(sample_val) > 40:
            sample_val = sample_val[:37] + "..."
        records.append({
            "column_name": col,
            "data_type": dtype,
            "null_count": null_count,
            "sample_value": sample_val,
            "description": descriptions.get(col, "Dataset attribute")
        })

    dict_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dict_df.to_csv(output_path, index=False)
    logger.info(f"Saved data dictionary to {output_path}")
    return dict_df

def extract_representative_sample(
    raw_train_path: str,
    output_sample_path: str,
    n_series: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Extract a representative sample of store-product time-series across all available dates.
    This preserves complete temporal continuity for each selected (store, product) pair.
    """
    if os.path.exists(output_sample_path):
        logger.info(f"Sample file already exists at {output_sample_path}. Loading existing sample.")
        return pd.read_parquet(output_sample_path)

    logger.info(f"Extracting representative sample of {n_series} time series from {raw_train_path}...")
    table = pq.read_table(raw_train_path, columns=["store_id", "product_id"])
    unique_pairs = table.to_pandas().drop_duplicates()
    
    np.random.seed(seed)
    selected_indices = np.random.choice(len(unique_pairs), size=min(n_series, len(unique_pairs)), replace=False)
    sampled_pairs = unique_pairs.iloc[selected_indices]
    
    # Read full columns for these selected series
    logger.info(f"Reading full records for {len(sampled_pairs)} selected series...")
    full_df = pd.read_parquet(raw_train_path)
    sampled_df = pd.merge(full_df, sampled_pairs, on=["store_id", "product_id"], how="inner")
    
    os.makedirs(os.path.dirname(output_sample_path), exist_ok=True)
    sampled_df.to_parquet(output_sample_path, index=False)
    logger.info(f"Saved sample ({len(sampled_df):,} rows) to {output_sample_path}")
    return sampled_df

def load_data(config_path: str = "config.yaml", use_sample: bool = True) -> pd.DataFrame:
    """Main data access function."""
    config = load_config(config_path)
    train_url = config["data"]["train_url"]
    eval_url = config["data"]["eval_url"]
    train_dest = config["data"]["train_parquet"]
    eval_dest = config["data"]["eval_parquet"]
    sample_dest = config["data"]["sample_parquet"]
    sample_size = config["data"]["sample_size_series"]

    # 1. Ensure raw datasets exist
    download_file(train_url, train_dest)
    download_file(eval_url, eval_dest)

    if use_sample:
        df = extract_representative_sample(train_dest, sample_dest, n_series=sample_size, seed=config["project"]["random_seed"])
    else:
        logger.info(f"Loading full training parquet from {train_dest}...")
        df = pd.read_parquet(train_dest)

    # 2. Inspect and generate data dictionary
    inspect_dataset(df, "Development Sample" if use_sample else "Full Train Dataset")
    create_data_dictionary(df)
    return df

if __name__ == "__main__":
    df = load_data()
    print("Data loading and inspection complete. Sample shape:", df.shape)
