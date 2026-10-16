# Telecom Customer Churn Prediction System

> Engineering review status: source and inference paths have been repaired and statically validated. The training notebook has **not** been rerun in this repository because the IBM Telco CSV is not bundled and heavy training dependencies are not installed in the audit environment.

## Overview

This project presents an end-to-end **Machine Learning system for Telecom Customer Churn Prediction**, designed to identify customers who are likely to leave a telecom service provider. The system enables proactive retention strategies and supports data-driven business decisions.

The solution contains a reproducible training notebook and a Streamlit inference UI. It is not a claim that the checked-in artifacts have been retrained from this checkout. It includes:
- Full data pipeline (cleaning → feature engineering → modeling → evaluation)
- Multiple machine learning models
- Hyperparameter optimization
- Explainability using SHAP
- Validation-set threshold tuning for business optimization (the test set remains untouched for final reporting)
- Profit-based business impact analysis
- Ready-to-deploy inference pipeline

---

## Problem Statement

Customer churn is one of the most critical challenges in the telecom industry. Losing customers directly impacts revenue and long-term business growth.

The objective of this system is to:
- Predict whether a customer will churn
- Estimate churn probability
- Prioritize high-risk customers
- Optimize retention campaign decisions

---

## Dataset

**Source:** IBM Telco Customer Churn Dataset  
**Reference dataset:** IBM Telco Customer Churn CSV (not included in this repository)
**Expected schema:** 7,043 rows and 20 original features when using the commonly distributed IBM file

Includes:
- Customer demographics
- Account information
- Service subscriptions
- Billing and payment details
- Target variable: `Churn`

---

## System Architecture

The system follows a production-style ML architecture:

Raw Data → Data Validation → Feature Engineering → Preprocessing Pipeline → ML Models → Model Selection → SHAP Explainability → Threshold Optimization → Business Decision Layer → Deployment

---

## Project Pipeline

This project is not a simple notebook workflow — it is designed as a **complete production-grade ML system** that simulates real-world telecom churn prediction in an enterprise environment.

The pipeline follows a strict ML lifecycle aligned with industry best practices:

### 1. Data Ingestion & Validation Layer
- Loading raw telecom customer data (IBM Telco Dataset)
- Schema inspection and data type validation
- Initial data integrity checks (missing values, duplicates, inconsistencies)
- Early detection of data quality issues before modeling

---

### 2. Exploratory Data Analysis (EDA) & Business Understanding
- Deep statistical analysis of customer behavior
- Churn distribution analysis (class imbalance detection)
- Feature-wise behavioral insights (tenure, charges, services)
- Correlation analysis and dependency mapping
- Business-driven insights extraction (not just visualization)

---

### 3. Data Cleaning & Preprocessing Engine
- Handling missing values with statistical imputation
- Correcting inconsistent categorical values (service noise normalization)
- Type casting and feature standardization
- Removing redundant identifiers and leakage-prone columns
- Target encoding (Churn → Binary format)

---

### 4. Advanced Feature Engineering Layer
Engineered features designed to simulate real telecom business intelligence:

- Customer Lifetime Value (CLV)
- Average Charge per Month
- Charge Growth / Inflation Ratio
- Customer Lifecycle Segmentation (New / Mid / Loyal)
- Service Adoption Index
- Risk Scoring Engine (Rule + Data driven hybrid)
- Behavioral Interaction Features (contract × payment × tenure)
- Polynomial transformations for non-linear relationships

---

### 5. Feature Selection & Dimensional Optimization
- Mutual Information based feature ranking
- Selection of most predictive variables
- Removal of noise and redundant signals
- Optimization of feature space for generalization

---

### 6. Machine Learning Preprocessing Pipeline (Production Style)
- ColumnTransformer-based architecture
- Separate handling for:
  - Numerical features (Scaling + Imputation)
  - Categorical features (One-Hot Encoding)
  - Binary features (Direct pass-through)
- Fully encapsulated preprocessing pipeline (deployment-ready)

---

### 7. Class Imbalance Handling Strategy
- Detection of severe class imbalance (~1:2.7 churn ratio)
- Model-level balancing using:
  - class_weight (Logistic Regression, Random Forest, LightGBM)
  - scale_pos_weight (XGBoost)
  - auto_class_weights (CatBoost)
- No data leakage SMOTE used inside pipeline design

---

### 8. Multi-Model Training Framework
A full benchmark suite of ML algorithms:

- Logistic Regression (Baseline + Interpretable)
- Random Forest (Bagging approach)
- XGBoost (Gradient Boosting)
- LightGBM (High-speed GBDT)
- CatBoost (Categorical-aware deep boosting)

All models are trained inside **Scikit-learn Pipelines**.

---

### 9. Robust Cross-Validation Strategy
- Stratified K-Fold Cross Validation (5 folds)
- ROC-AUC as primary evaluation metric
- Stability analysis across folds
- Variance monitoring for overfitting detection

---

### 10. Hyperparameter Optimization (Optuna Engine)
- Bayesian Optimization using Optuna
- Search space tuning for CatBoost
- Automated performance maximization
- Multi-trial convergence analysis

---

### 11. Model Evaluation & Ranking System
Comprehensive evaluation using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Precision-Recall AUC

Final ranking based on business-aligned performance, not just accuracy.

---

### 12. Explainable AI Layer (SHAP Integration)
Full interpretability framework:

- Global feature importance (model-wide behavior)
- Local explanations (per-customer prediction)
- SHAP summary plots
- Waterfall analysis for churn cases
- Business translation of model decisions

---

### 13. Probability Calibration Analysis
- Reliability curve evaluation
- Brier Score computation
- Calibration quality assessment
- Ensuring probabilistic trustworthiness for decision systems

---

### 14. Decision Threshold Optimization Engine
- Moving beyond default 0.5 threshold
- Optimization using F2-Score (recall-focused business logic)
- Trade-off analysis between precision and recall
- Business-aligned decision boundary selection

---
## Model Evaluation & Final Benchmark

The historical benchmark table below is retained as project context only; it is not recomputed by the current checkout. Do not treat it as a newly verified result until the notebook is rerun against a known dataset snapshot and dependency lockfile.

- Best Model: Tuned CatBoost
- Evaluation Metric: ROC-AUC
- Threshold Optimization: Improved Recall significantly
- Business Impact: Positive ROI achieved

---

The following table summarizes the performance of all trained machine learning models based on cross-validation ROC-AUC and test set evaluation:

| Model                | ROC-AUC | Strengths                          | Business Impact                        |
|---------------------|----------|------------------------------------|----------------------------------------|
| Logistic Regression | 0.8507   | Fast, interpretable                | Real-time baseline decision system     |
| Random Forest       | 0.8421   | Robust non-linearity handling      | Stable segmentation model              |
| LightGBM            | 0.8346   | Fast gradient boosting             | Scalable production model              |
| CatBoost            | 0.8400   | Handles categorical data well      | Strong general-purpose predictor       |
| XGBoost             | 0.8355   | High-performance boosting          | Competitive but slightly lower recall  |
| CatBoost (Tuned)    | BEST     | Optimized via Optuna               | Final production decision engine       |

---

### Final Selected Model
The notebook is configured to select and export the tuned CatBoost pipeline, but the checked-in artifact should be regenerated after the repaired notebook is run. **CatBoost (Tuned)** was selected as the production model due to:
- Best balance between Precision and Recall
- Strong generalization performance
- Stability under cross-validation
- Superior business impact after threshold tuning

---

## 15. Key Business Insights

1. **Contract Type is the strongest churn driver**
   - Month-to-month customers have the highest churn probability

2. **Customer Tenure is critical**
   - First 12 months represent highest risk period

3. **Fiber optic users churn more**
   - Indicates service expectation or quality issues

4. **Lack of add-on services increases churn**
   - Security and tech support reduce churn significantly

5. **Payment method affects retention**
   - Electronic check users are most likely to churn

---

## Business Impact Analysis

The system is evaluated using a profit-based cost model:

- Revenue per customer: $65/month
- Annual value per retained customer: $780
- Retention campaign cost: $20

### Outcome:
- Significant reduction in churn loss
- Increased targeted retention efficiency
- Positive ROI compared to no-model scenario
- High-value customer retention improvement

---

## 16.Deployment

The repository contains a deployment-oriented reference implementation, but it is **not certified production-ready from this checkout** because the bundled artifacts are legacy outputs and the training dataset is intentionally absent. Before deployment, regenerate and validate the artifacts with a known dataset snapshot and resolved environment. The implementation includes:

- Trained ML pipeline (`joblib`)
- Feature engineering function
- Threshold configuration
- Prediction function API
 
---

### Example Usage

Input:
- Customer profile dictionary

Output:
- Churn probability
- Risk level (LOW / MEDIUM / HIGH)
- Business recommendation

---

## Prediction Logic

The system returns:
- `churn_prediction` (True/False)
- `churn_probability`
- `risk_level`
- `recommendation`

Example:
- HIGH risk → Immediate retention action
- MEDIUM risk → Monitoring
- LOW risk → No action required

---

## Project Output

- End-to-end ML pipeline
- Business-driven feature engineering
- Optimized classification model
- Explainable AI layer (SHAP)
- Profit-aware decision system
- Deployable production artifact

---

## Conclusion

This system represents a **complete AI-driven churn prediction and retention intelligence engine** that goes beyond traditional machine learning.

It transforms raw telecom data into actionable business decisions by combining:

- Machine Learning
- Feature Engineering
- Explainable AI
- Business Intelligence
- Profit Optimization

The final result is a deployable system capable of delivering measurable financial impact and improving customer retention strategies in real-world telecom environments.


## Reproducible setup and data

1. Create an environment with Python 3.10+ and install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Obtain the IBM Telco Customer Churn CSV separately. The repository intentionally does not fabricate or redistribute training data.

3. Set the input path before running the notebook:

   ```bash
   export CHURN_DATA_PATH=/absolute/path/to/churn_prediction.csv
   jupyter notebook "Telecom Customer Churn Prediction System.ipynb.ipynb"
   ```

   If the variable is not set, the notebook looks for `data/churn_prediction.csv` relative to the notebook working directory and raises an actionable `FileNotFoundError` when it is absent.

The repaired notebook uses a stratified train/validation/test split. Numeric imputation is learned inside the training pipeline, threshold selection uses validation data, and the test split is used only for final evaluation/business-impact reporting. Existing serialized artifacts are legacy outputs from the original project; because no training data is bundled, they are not silently replaced during this review.

## Streamlit inference

```bash
streamlit run app.py
```

The app resolves artifacts relative to `app.py`, validates required input fields and the configured threshold, and reports a clear error if artifacts are missing or incompatible.

## Canonical input contract

`churn_features.py` is the shared implementation used by the notebook and Streamlit app. It validates the 19 raw IBM Telco feature columns, parses numeric fields without fitting or imputing on inference data, allows blank `TotalCharges` for the training-only imputer, rejects unsupported categories/invalid values, and locates the positive probability by the artifact's class label rather than assuming a column position. Focused contract tests are in `tests/test_feature_contract.py`.

The repaired notebook writes plots to `CHURN_OUTPUT_DIR` (default: current working directory) and artifacts to `CHURN_ARTIFACT_DIR` (default: `artifacts/`). Its model config records the input dataset SHA-256 and feature-contract version. Existing checked-in artifacts do not contain those new metadata fields and should be treated as legacy until regenerated.
