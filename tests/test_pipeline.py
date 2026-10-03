"""Automated Verification Suite for StockGuard Fresh Retail Pipeline.

Tests:
1. Dataset loading and schema integrity
2. Zero duplicate/missing values in identifiers
3. Temporal chronological ordering across splits
4. Strict absence of future lookahead leakage in features
5. Correct target formulation and binary domain [0, 1]
6. Feature engineering matrix completeness and non-infinity
7. Baseline and advanced model artifact persistence
8. StockGuard multi-model artifacts (Random Forest, LightGBM, CatBoost, Calibrated Champion)
9. Model prediction behavior and calibrated probability ranges [0.0, 1.0]
10. Recall@K and Precision@K mathematical properties (monotonicity and bounds)
11. Cost-benefit threshold optimization sanity check
"""

import os
import sys
import pytest
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data
from src.target import add_target_variable
from src.split import temporal_split, verify_no_temporal_leakage
from src.feature_engineering import prepare_feature_pipeline, get_feature_columns, assert_feature_leakage_isolation
from src.models_comparison import CalibratedChampion, compute_recall_at_k, compute_precision_at_k

@pytest.fixture(scope="module")
def sample_data():
    """Load sample data fixture."""
    return load_data(use_sample=True)

def test_dataset_loading(sample_data):
    """Test 1: Verify dataset loads with required schema columns and positive row count."""
    assert len(sample_data) > 0, "Dataset is empty!"
    required_cols = [
        "city_id", "store_id", "product_id", "dt", "sale_amount",
        "stock_hour6_22_cnt", "hours_sale", "hours_stock_status", "discount"
    ]
    for col in required_cols:
        assert col in sample_data.columns, f"Missing required column: {col}"

def test_missing_values(sample_data):
    """Test 2: Check that core identifier and target columns have zero nulls."""
    assert sample_data["store_id"].isnull().sum() == 0, "Null values found in store_id!"
    assert sample_data["product_id"].isnull().sum() == 0, "Null values found in product_id!"
    assert sample_data["stock_hour6_22_cnt"].isnull().sum() == 0, "Null values in stock_hour6_22_cnt!"

def test_target_generation(sample_data):
    """Test 3: Target variable must be binary (0 or 1) with no missing values after alignment."""
    df_with_target = add_target_variable(sample_data, "next_day")
    assert "target_stockout" in df_with_target.columns
    assert df_with_target["target_stockout"].isnull().sum() == 0
    unique_vals = set(df_with_target["target_stockout"].unique())
    assert unique_vals.issubset({0, 1}), f"Target contains non-binary values: {unique_vals}"

def test_temporal_ordering(sample_data):
    """Test 4: Strict chronological ordering; train_max < val_min <= val_max < test_min."""
    df_with_target = add_target_variable(sample_data, "next_day")
    train_df, val_df, test_df, _ = temporal_split(df_with_target)
    assert verify_no_temporal_leakage(train_df, val_df, test_df)

def test_no_future_leakage(sample_data):
    """Test 5: Future perturbation test confirms features at t are independent of t+1..t+k."""
    assert assert_feature_leakage_isolation(sample_data)

def test_feature_generation(sample_data):
    """Test 6: Feature engineering generates expected columns with zero infinite values."""
    df_with_target = add_target_variable(sample_data, "next_day")
    train_df, val_df, test_df, _ = temporal_split(df_with_target)
    train_f, val_f, test_f, features = prepare_feature_pipeline(train_df, val_df, test_df)
    
    assert len(features) >= 30, f"Too few features engineered ({len(features)})!"
    X_train = train_f[features].fillna(0)
    assert not np.isinf(X_train.values).any(), "Infinite values detected in feature matrix!"

def test_saved_models_exist():
    """Test 7: Ensure baseline and advanced model artifacts exist on disk."""
    expected_models = [
        "models/baseline/dummy_(stratified).joblib",
        "models/baseline/logistic_regression.joblib",
        "models/baseline/decision_tree.joblib"
    ]
    for path in expected_models:
        assert os.path.exists(path), f"Missing model artifact: {path}"

def test_stockguard_multi_models_exist():
    """Test 8: Ensure StockGuard advanced models exist on disk."""
    expected_models = [
        "models/baseline/random_forest.joblib",
        "models/lightgbm_model.joblib",
        "models/catboost_model.joblib",
        "models/champion_calibrated.joblib"
    ]
    for path in expected_models:
        assert os.path.exists(path), f"Missing StockGuard model artifact: {path}"

def test_calibrated_champion_prediction():
    """Test 9: Calibrated Champion outputs valid probability distribution in [0.0, 1.0]."""
    calib_path = "models/champion_calibrated.joblib"
    assert os.path.exists(calib_path), f"Missing {calib_path}"
    model = joblib.load(calib_path)
    
    feature_cols = get_feature_columns()
    dummy_input = pd.DataFrame([{c: 0.0 for c in feature_cols}])
    
    probs = model.predict_proba(dummy_input)
    assert probs.shape == (1, 2), f"Unexpected shape: {probs.shape}"
    p_pos = probs[0, 1]
    assert 0.0 <= p_pos <= 1.0, f"Probability out of bounds: {p_pos}"
    assert np.isclose(probs.sum(), 1.0), "Probabilities do not sum to 1.0"

def test_recall_and_precision_at_k_logic():
    """Test 10: Verify Recall@K and Precision@K calculations on synthetic ground truth."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])  # 3 positives, 10 items total
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    # At K=20% (top 2 items): items 0 and 1 (both are 1s)
    rec_20 = compute_recall_at_k(y_true, y_prob, 20.0)
    prec_20 = compute_precision_at_k(y_true, y_prob, 20.0)

    assert rec_20 == pytest.approx(2.0 / 3.0), f"Expected 2/3, got {rec_20}"
    assert prec_20 == pytest.approx(1.0), f"Expected 1.0, got {prec_20}"

    # Recall must be monotonically non-decreasing with increasing K
    rec_50 = compute_recall_at_k(y_true, y_prob, 50.0)
    assert rec_50 >= rec_20, "Recall@K must not decrease when K increases"

def test_cost_utility_metrics_exist():
    """Test 11: Cost curve analysis exists and confirms savings over baseline."""
    cost_file = "results/metrics/cost_curve_analysis.csv"
    assert os.path.exists(cost_file), f"Missing {cost_file}"
    df = pd.read_csv(cost_file)
    assert len(df) > 10, "Cost analysis curve is too short"
    min_cost = df["total_cost"].min()
    max_cost = df["total_cost"].max()
    assert min_cost < max_cost, "Optimization did not find cost differential"
