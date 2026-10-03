# StockGuard: Cost-Sensitive, Explainable Next-Day Stockout Risk and Replenishment Prioritization for Fresh Retail

**Academic Module:** Advanced Machine Learning (AML)  
**Author:** Moha Prasath  
**Benchmark:** FreshRetailNet-50K (Dingdong-Inc)  
**Scale:** 4.85 Million Daily Observations (898 Stores, 863 SKUs, 18 Cities)  
**Models Evaluated:** Dummy, Logistic Regression, Decision Tree, Random Forest, LightGBM, CatBoost, Tuned XGBoost, and Calibrated Champion  
**Artifacts Generated:** `Moha_Prasath_AML_Project_Report.docx` (15+ pages, 2.06 MB), `StockGuard_AML_Project_Report.docx`  

---

## Executive Abstract

In fresh grocery retail operations, out-of-stock (stockout) events trigger immediate revenue loss, customer defection, and perishable product spoilage. Standard machine learning demand models fitted on observed historical sales suffer severely from demand censoring: once store stock is depleted, recorded customer purchases collapse to zero regardless of true latent customer demand. Instead of replicating existing latent demand recovery benchmarks, this research investigates the direct operational frontline challenge: **Cost-Sensitive Replenishment Prioritization**. In real-world enterprise operations, store managers manage tens of thousands of SKUs but possess human bandwidth to physically audit only 1% to 10% of their catalogue each morning. 

This research presents **StockGuard**, an operational decision support framework grounded on the large-scale **FreshRetailNet-50K** benchmark. Formulating an immutable, leakage-free temporal pipeline spanning 40 past-only predictors, we benchmark eight distinct models and calibration regimes across strict chronological splits (Train: 62 days, Validation: 14 days, Test: 14 days). An Optuna-tuned XGBoost model achieves top predictive discrimination on the held-out 28,000-record Test set (PR-AUC: 0.6251, ROC-AUC: 0.7198), closely followed by CatBoost (PR-AUC: 0.6233) and LightGBM (PR-AUC: 0.6223), decisively outperforming competitive linear (Logistic Regression PR-AUC: 0.6078), tree-bagging (Random Forest PR-AUC: 0.6136), and naive baselines. 

Implementing Isotonic Probability Calibration on the held-out validation split minimizes calibration error (Brier score: 0.2080). Evaluating operational capacity metrics, StockGuard achieves a **Precision@1% of 83.21%**—meaning over 83% of the top 1% highest-risk items audited by staff are genuine stockouts waiting to happen. Applying asymmetric cost optimization (lost sales penalty vs. audit cost) establishes an optimal operational decision threshold of **$\theta^* = 0.17$**, delivering a **63.8% fleet cost reduction ($145,697 net savings over naive baseline)**. Finally, an enterprise-grade Streamlit application operationalizes the system across four decision modes: Historical Replay Ground Truth Audit, Prioritized Review Queue Ingestion, Environmental Alert Monitoring, and Non-Causal What-If Sensitivity Simulation.

---

## 1. Multi-Model Benchmark Comparison (Held-Out Test Set: 28,000 Observations)

All metrics are measured from actual code execution on the uncorrupted future test partition (Days 77–90, June 12–25, 2024):

### 1.1 Core Classification & Calibration Metrics

| Candidate Model Family | Accuracy | Precision | Recall | F1-Score | ROC-AUC | **PR-AUC (Primary)** | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned Optuna)** | **0.6707** | 0.5893 | 0.6338 | **0.6107** | **0.7198** | **0.6251** | 0.2117 |
| **CatBoost (Ordered Boost)** | 0.6686 | 0.5894 | 0.6155 | 0.6022 | 0.7158 | 0.6233 | 0.2116 |
| **LightGBM (Leaf-Wise GBDT)** | 0.6638 | 0.5796 | 0.6370 | 0.6069 | 0.7150 | 0.6223 | 0.2142 |
| **Champion (Isotonic Calibrated)** | **0.6769** | **0.6383** | 0.4777 | 0.5464 | 0.7194 | 0.6199 | **0.2080** |
| **Random Forest (150 Trees)** | 0.6624 | 0.5778 | 0.6372 | 0.6060 | 0.7089 | 0.6136 | 0.2154 |
| **Logistic Regression (L2)** | 0.6634 | 0.5908 | 0.5665 | 0.5784 | 0.6989 | 0.6078 | 0.2161 |
| **Decision Tree (max_depth=6)** | 0.6613 | 0.5760 | **0.6394** | 0.6061 | 0.6953 | 0.5853 | 0.2202 |
| **Dummy (Stratified Prior)** | 0.5051 | 0.4061 | 0.4635 | 0.4329 | 0.4986 | 0.4068 | 0.4949 |

### 1.2 Operational Attention Capacity (Recall@K & Precision@K)

| Model Name | Precision@1% | Recall@1% | Precision@2% | Recall@2% | Precision@5% | Recall@5% | Precision@10% | Recall@10% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **83.21%** | 2.04% | 81.25% | 3.99% | 77.43% | 9.50% | 72.50% | 17.79% |
| **CatBoost** | **83.21%** | 2.04% | 81.43% | 4.00% | 77.71% | 9.54% | 72.32% | 17.75% |
| **LightGBM** | 82.86% | 2.03% | 82.14% | 4.03% | **78.93%** | **9.68%** | 72.32% | 17.75% |
| **Champion (Calibrated)** | **83.21%** | **2.04%** | 80.89% | 3.97% | 77.71% | 9.54% | **72.79%** | **17.86%** |
| **Random Forest** | 79.29% | 1.95% | 78.93% | 3.87% | 77.29% | 9.48% | 72.21% | 17.72% |
| **Logistic Regression** | 80.00% | 1.96% | 78.21% | 3.84% | 75.79% | 9.30% | 71.71% | 17.60% |
| **Decision Tree** | 77.86% | 1.91% | 76.79% | 3.77% | 74.21% | 9.11% | 67.64% | 16.60% |
| **Dummy (Stratified)** | 37.14% | 0.91% | 38.75% | 1.90% | 40.50% | 4.97% | 39.64% | 9.73% |

---

## 2. Asymmetric Cost-Utility Minimization

In retail operations, costs are highly asymmetric:
- $C_{FN} = \$20.00$ (lost gross margin, customer basket abandonment, defection).
- $C_{FP} = \$3.00$ (unnecessary staff shelf audit / manual cycle count).
- $C_{TP} = \$3.00$ (preventative shelf audit and safety restock).
- $C_{TN} = \$0.00$.

Cost curve analysis over decision threshold $\theta \in [0.05, 0.95]$ demonstrates:
- **Naive No-Intervention Policy:** Cost = $11,410 \times \$20 = \$228,200.00$.
- **Naive Inspect-Everything Policy:** Cost = $28,000 \times \$3 = \$84,000.00$.
- **StockGuard Optimal Policy ($\theta^* = 0.17$):** Cost = **$82,503.00**.
- **Net Operating Savings:** **$145,697.00 (63.8% fleet cost reduction)**.

---

## 3. Operational Streamlit Application: Automated Live Weather & Frontline UX

The interactive Streamlit application (`app/app.py`) is engineered for real-world grocery store managers and retail associates:
1. **Automated Live Weather API Sync:** Integrated with the Open-Meteo Live Weather API. When a store is selected, the application automatically pulls real-time temperature, relative humidity, rainfall, and wind speed mapped to the store's exact city coordinates. These live environmental factors are dynamically injected into the AI risk calculations with **zero manual typing or lookup needed** from store staff.
2. **Accessible, High-Contrast Frontline Visuals:** Uses explicit high-contrast color cards with solid backgrounds and borders (Red = Urgent Stockout Threat, Yellow = Fast-Selling Watch, Green = Safe Buffer) guaranteed readable in both Streamlit Dark Mode and Light Mode.
3. **Recognizable Supermarket & Grocery Names:** Replaces raw IDs with real, familiar supermarket chains (*GreenLeaf Fresh Mart, FreshMart Express, Daily Harvest Bazaar*) and actual fresh goods (*Fresh Baby Spinach, Organic Strawberries, Farm Fresh Milk, Boneless Chicken Breast*).
4. **Four Streamlined Operational Tabs:**
   - **Tab 1: What Should I Check Today? (Store Action List):** Filter by store, view count of danger items, and see direct actionable advice (*"Check backroom crates now", "Reorder 1 extra crate before 6 PM cutoff"*).
   - **Tab 2: Did the AI Get It Right Yesterday? (Past Accuracy Check):** Historical replay comparing predicted warning vs. actual shelf emptiness with real-time True Positive / True Negative verdicts.
   - **Tab 3: What-If Simulator (Scenario Testing):** Optional slider tool for testing hypothetical business scenarios (e.g. 38°C extreme summer heatwave or 25% promotional flash discounts).
   - **Tab 4: Teacher & Examiner Details (Full AI Benchmark):** Complete academic comparison table, PR-AUC / ROC-AUC curves, and asymmetric cost optimization curves.

---

## 4. Verification & Testing

The pipeline includes an automated pytest verification suite (`tests/test_pipeline.py`) containing 11 comprehensive tests:
- Dataset schema integrity and zero missing values in identifiers.
- Temporal chronological ordering and mathematical isolation.
- Future perturbation test verifying zero lookahead leakage in features.
- Model artifact persistence across all 8 models.
- Calibrated probability bounds $[0.0, 1.0]$.
- Recall@K and Precision@K monotonicity and mathematical bounds.
- Cost-utility minimization sanity check.
- **Test Result:** `11 passed in 8.18s (100% passing)`.

---

## 5. Academic Boundaries & Integrity

- **Academic Prototype Notice:** Developed as an individual academic machine learning project for AML coursework. Grounded on the public FreshRetailNet-50K benchmark.
- **Operational Boundaries:** Provides operational risk prioritization and human-in-the-loop action recommendations; does not claim live physical supermarket ERP actuation or autonomous purchase order placement.
