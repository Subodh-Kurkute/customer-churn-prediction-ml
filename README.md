# Customer Churn Prediction using Machine Learning

## 📌 Project Overview
Customer churn prediction is a critical business problem for subscription-based companies, where retaining existing customers is often more cost-effective than acquiring new ones. This project aims to predict customer churn using historical customer data and to identify key drivers influencing churn behavior.

The project follows a systematic, industry-aligned machine learning workflow, progressing from interpretable baseline models to advanced ensemble methods, with a strong emphasis on evaluation metrics aligned to business objectives.

---

## 🎯 Problem Statement
The objective is to build a classification model that can accurately identify customers who are likely to churn.  
Given the business context, **recall is prioritized** to minimize missed churners, while maintaining reasonable precision to control unnecessary retention efforts.

---

## 📊 Dataset
- **Source**: IBM Telco Customer Churn dataset (Kaggle / OpenML)
- **Size**: ~7,000 customers
- **Features**:
  - Demographics (e.g., gender, senior citizen, dependents)
  - Services subscribed (internet, security, streaming, support)
  - Contract and billing information
  - Tenure and pricing details
- **Target Variable**: `Churn` (Yes / No)

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was conducted in a hypothesis-driven manner, focusing on:
- Churn distribution and class imbalance
- Relationship between churn and tenure, contract type, pricing, and services
- Aggregated service usage patterns
- Geographic features (explored and discarded due to lack of churn signal)

Key insights included:
- Short tenure and month-to-month contracts strongly increase churn risk
- Higher monthly charges are associated with higher churn
- Customers with more engaged service usage churn less
- Geographic clustering showed no meaningful correlation with churn

---

## 🧹 Data Preprocessing
Key preprocessing steps:
- Dropped identifier and non-informative geographic features
- Handled missing values in numerical fields (e.g., `Total Charges`)
- One-hot encoded categorical variables
- Separate preprocessing pipelines were used for:
  - **Linear models** (scaling + transformations)
  - **Tree-based models** (no scaling)
- A preprocessing correction was applied to ensure numerical features were properly passed to tree-based models

This correction significantly improved Random Forest and XGBoost performance and is documented transparently in the analysis.

---

## 🤖 Models Evaluated
The following models were trained and evaluated:

- Logistic Regression (baseline)
- Logistic Regression (threshold optimized)
- Random Forest
- Random Forest (threshold optimized)
- XGBoost (final model)

Threshold tuning was applied where appropriate to optimize recall.

---

## 📈 Model Performance (Validation)

| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| Logistic Regression | 0.818 | 0.686 | 0.576 | 0.626 |
| Logistic Reg (Opt. Threshold) | 0.798 | 0.596 | 0.746 | 0.663 |
| Random Forest | 0.789 | 0.570 | 0.836 | 0.678 |
| RF (Opt. Threshold) | 0.804 | 0.597 | 0.803 | 0.685 |
| **XGBoost** | **0.844** | **0.641** | **0.934** | **0.760** |

---

## 🧪 Final Test Set Performance

| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| **XGBoost (Test)** | 0.754 | 0.526 | 0.757 | 0.621 |

While a generalization drop was observed (expected due to recall-focused tuning), the model maintained strong recall on unseen data, confirming robust performance.

---

## ✅ Model Selection Rationale
- Logistic Regression provided a strong, interpretable baseline but plateaued due to linear assumptions.
- Random Forest improved performance once numerical features were correctly included but offered incremental gains.
- XGBoost consistently achieved the highest recall and F1 score by capturing non-linear interactions and focusing on hard-to-classify churn cases.
- Given the business priority of maximizing churn detection, **XGBoost was selected as the final model**.

---

## ⚠️ Limitations
- Static snapshot data; no temporal behavior modeling
- Threshold optimized on validation data
- Single-provider, single-region dataset
- No explicit cost-sensitive optimization
- Model probabilities not calibrated for deployment

---

## 🚀 Future Improvements
- Incorporate time-series and behavioral features
- Cross-validated threshold optimization
- Cost-aware churn modeling
- Probability calibration for production use
- Deployment and monitoring pipeline

---

## 🧠 Key Takeaways
- Proper preprocessing is critical for fair model comparison
- Threshold tuning significantly impacts churn recall
- Boosting models outperform bagging and linear models for structured churn data
- Recall-focused evaluation aligns better with business objectives

---

## 📎 Tools & Libraries
- Python, pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib, Seaborn

---

## 👤 Author
This project was built as part of a machine learning portfolio to demonstrate end-to-end problem solving, model selection, and business-aligned evaluation.


