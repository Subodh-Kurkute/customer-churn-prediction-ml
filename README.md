# Telecom Customer Churn Prediction

## Overview
Customer churn is a critical business problem for subscription-based companies, where retaining existing customers is often more cost-effective than acquiring new ones. This project focuses on predicting customer churn in the telecom domain using supervised machine learning, with an emphasis on recall-focused evaluation, interpretability, and business realism.

The project follows an end-to-end ML workflow, progressing from exploratory data analysis and hypothesis-driven feature exploration to model comparison, threshold optimization, and interpretability analysis.

---

## Problem Statement
The objective of this project is to predict whether a customer will churn (leave the service) based on demographic, contractual, pricing, and service usage information.

- **Target Variable:** `Churn Label`
- **Problem Type:** Binary classification
- **Business Goal:** Maximize churn detection (recall) to enable proactive retention strategies

---

## Dataset
The project uses the **IBM Telco Customer Churn dataset**, sourced via Kaggle.

- Source:  
  https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset
- Original IBM documentation:  
  https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

**Note:**  
The raw dataset is not included in this repository due to licensing considerations. Instructions for downloading the dataset are provided within the notebook.

---

## Project Highlights

- End-to-end churn prediction project with realistic business framing.
- Hypothesis-driven EDA guiding feature selection and engineering decisions.
- Separate preprocessing pipelines for linear and tree-based models.
- Threshold optimization to align model predictions with recall-focused churn objectives.
- Comparative evaluation of Logistic Regression, Random Forest, and XGBoost.
- Model interpretability using SHAP for both global and local explanations.

---

## Methodology

### 1. Exploratory Data Analysis
- Hypothesis-based exploration of demographics, tenure, contracts, pricing, and services.
- Identification and removal of redundant, high-cardinality, and leakage-prone features.
- Validation of business assumptions using churn rates and distributional analysis.

### 2. Feature Engineering
- Aggregation of individual service indicators into a service engagement feature.
- Careful handling of missing values and skewed numerical features.
- Explicit exclusion of post-outcome variables (e.g., churn reason, churn score).

### 3. Modeling
Models evaluated include:
- Logistic Regression (baseline and threshold-optimized)
- Random Forest (default and threshold-optimized)
- XGBoost (final selected model)

Separate preprocessing pipelines were designed for:
- Linear models (scaling and log transformations)
- Tree-based models (no scaling or log transformations)

### 4. Evaluation Strategy
- Primary focus on **Recall** and **F1-score** due to class imbalance and business cost of missed churners.
- Threshold optimization performed using precision–recall trade-offs.
- Final evaluation conducted on a held-out test set.

---

## Model Selection Summary
Logistic Regression provided a strong and interpretable baseline, with threshold optimization significantly improving recall. Random Forest models offered limited incremental benefit over the tuned linear baseline. XGBoost demonstrated the best overall performance, achieving the highest recall and F1-score while maintaining acceptable precision, making it the most suitable model for the churn detection objective.

---

## Interpretability
- Gain-based feature importance for model-level insights.
- SHAP summary and waterfall plots for global and local explanations.
- Interpretability analysis confirms that churn is driven primarily by contract structure, tenure, pricing pressure, and service engagement rather than demographics.

---

## Key Business Insights

- Contract type is the strongest churn driver; month-to-month customers churn significantly more.
- New customers are more vulnerable to churn, highlighting the importance of early engagement.
- Higher monthly charges increase churn risk even after accounting for tenure.
- Service quality and protection features (e.g., tech support, online security) reduce churn probability.
- Demographic attributes contribute minimal predictive value.

---

## Limitations
- The analysis is based on a static snapshot of customer data without temporal dynamics.
- Threshold optimization may not generalize optimally under distributional shifts.
- Results are specific to a single telecom dataset and geographic context.

---

## Future Scope
- Incorporate time-based features and longitudinal modeling for churn dynamics.
- Explore cost-sensitive learning and custom loss functions aligned with business impact.
- Investigate controlled use of auxiliary signals (e.g., churn reasons, CLTV) via weak supervision or multi-task learning while avoiding data leakage.

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Reproducibility
All experiments use fixed random seeds where applicable to ensure reproducibility. The notebook is designed to run end-to-end once the dataset is downloaded.

---

## Author
This project was developed as part of a professional machine learning portfolio, with an emphasis on real-world problem framing, evaluation rigor, and interpretability.
