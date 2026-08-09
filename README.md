<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
</p>

# 📉 Telco Customer Churn Prediction

> **An end-to-end machine learning project** that predicts which telecom customers are likely to churn, estimates revenue at risk, and generates smart, personalized retention strategies — all from a single notebook.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Challenges & How We Tackled Them](#-challenges--how-we-tackled-them)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Customer Retention System](#-customer-retention-system)
- [Revenue Impact Estimator](#-revenue-impact-estimator)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Key Takeaways](#-key-takeaways)
- [Future Improvements](#-future-improvements)

---

## 🎯 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Acquiring a new customer costs **5–7x more** than retaining an existing one. This project aims to:

1. **Predict** which customers are most likely to leave.
2. **Quantify** the revenue at risk from potential churners.
3. **Recommend** personalized retention actions based on customer value and churn probability.

---

## 📊 Dataset

- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 21 columns (demographics, account info, services subscribed)
- **Target:** `Churn` (Yes / No) — approximately **73% No, 27% Yes**

| Feature Category | Examples |
|---|---|
| **Demographics** | Gender, SeniorCitizen, Partner, Dependents |
| **Account Info** | Tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges |
| **Services** | PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, etc. |

---

## 🔄 Project Workflow

The notebook is structured into **4 numbered parts** followed by two standalone business-intelligence sections:

```
Part 1 → Data Inspection & Quality Assessment
Part 2 → Exploratory Data Analysis (EDA)
Part 3 → Data Cleaning & Preprocessing  →  saves processed_telco_churn.csv
Part 4 → Model Training & Evaluation    ←  reads processed_telco_churn.csv
         ├── Customer Retention System
         └── Customer Revenue Estimator
```

> **Note:** `processed_telco_churn.csv` is the cleaned and encoded dataset produced by Part 3 and consumed by Part 4. It is also included in this repo for standalone use.

---

## 🧩 Challenges & How We Tackled Them

### 1. Imbalanced Dataset

The dataset has roughly a **73/27 split** between non-churn and churn customers.

**Problem:** The model would be biased towards predicting "No Churn" since the majority class dominates.

**Solution:**
- Applied **SMOTE (Synthetic Minority Over-Sampling Technique)** to generate synthetic samples for the minority class during training.
- Used **`class_weight='balanced'`** in the Random Forest classifier to penalize misclassifying the minority class more heavily.
- Evaluated with **F1-Score and ROC-AUC** instead of just accuracy, since accuracy is misleading on imbalanced data.

---

### 2. Hidden Missing Values in `TotalCharges`

**Problem:** The `TotalCharges` column was stored as a string (`object` dtype) instead of numeric. It contained whitespace entries (`" "`) that pandas interpreted as valid values, so `isnull()` reported **zero missing values** — a silent data quality trap.

**Solution:**
- Used `pd.to_numeric(df['TotalCharges'], errors='coerce')` to force conversion, which turned non-parseable strings into `NaN`.
- Filled the resulting NaN values with the **median** of `TotalCharges` to avoid skewing by outliers.

---

### 3. High Cardinality in Categorical Features

**Problem:** After one-hot encoding all categorical features, the feature space expanded significantly, risking overfitting and increased training time.

**Solution:**
- Used `pd.get_dummies(df, drop_first=True)` to drop the first dummy column per category (avoiding the dummy variable trap and reducing dimensions).
- Applied **RobustScaler** in the Logistic Regression pipeline to handle feature scale differences without being sensitive to outliers.

---

### 4. Hyperparameter Tuning Complexity

**Problem:** Logistic Regression has multiple interacting hyperparameters (`C`, `penalty`, `solver`, `class_weight`), and certain combinations are incompatible (e.g., `l1` penalty doesn't work with all solvers).

**Solution:**
- Used **GridSearchCV** with a carefully designed parameter grid that only includes compatible solver-penalty combinations (`liblinear` and `saga`).
- Combined with **StratifiedKFold (5 splits)** to ensure each fold maintains the same class distribution as the full dataset.
- Optimized on **F1-Score** rather than accuracy.

---

### 5. Feature Engineering for Business Value

**Problem:** Raw features alone don't capture customer value or spending behavior well enough for business decisions.

**Solution:**
- Engineered **`AverageMonthlySpend`** = `TotalCharges / (tenure + 1)` to normalize spending across tenure lengths.
- Engineered **`TotalServices`** — a count of how many services each customer subscribes to (PhoneService, OnlineSecurity, TechSupport, StreamingTV, etc.).
- Engineered **`CLV (Customer Lifetime Value)`** = `MonthlyCharges × tenure` for retention prioritization.

---

### 6. Building an Actionable Retention System

**Problem:** A churn prediction score alone isn't useful to business teams. They need to know **who to act on**, **how urgently**, and **what to offer**.

**Solution:**
Built a multi-layered retention engine:
- **Priority Score (0–100):** Weighted combination of churn probability (70%) and normalized CLV (30%).
- **Priority Levels:** Critical (≥80), High (≥60), Medium (≥40), Low (<40).
- **Smart Retention Actions:** Rule-based system that considers priority level, CLV, tenure, and monthly charges to recommend specific offers:

| Priority | Condition | Retention Action |
|---|---|---|
| 🔴 Critical | CLV ≥ 5000 | Immediate Call + 30% Discount + Dedicated Manager |
| 🔴 Critical | Tenure < 12 | Free 2-Month Subscription + Onboarding Support |
| 🟠 High | CLV ≥ 5000 | Premium Support + Service Upgrade |
| 🟡 Medium | Tenure < 12 | Welcome Offer + Personalized Email |
| 🟢 Low | CLV ≥ 5000 | Loyalty Rewards + Thank You Coupon |

---

## 📈 Exploratory Data Analysis

The EDA phase used **Plotly (dark theme)** and **Seaborn** for interactive and static visualizations:

| Visualization | Key Insight |
|---|---|
| **Churn Distribution** | ~27% of customers churned — significant imbalance |
| **Tenure vs Churn** | Customers with shorter tenure churn more frequently |
| **Monthly Charges vs Churn** | Higher monthly charges correlate with higher churn |
| **Contract Type vs Churn** | Month-to-month contracts have drastically higher churn rates |
| **Internet Service vs Churn** | Fiber optic users churn more than DSL users |
| **Online Security / Tech Support vs Churn** | Customers without these services churn significantly more |
| **Payment Method vs Churn** | Electronic check payers have the highest churn rate |
| **Correlation Heatmap** | Tenure negatively correlated with churn; MonthlyCharges positively correlated |
| **Total Services vs Churn** | Customers using fewer services are more likely to churn |

---

## 🤖 Model Training & Evaluation

### Models Built

| Model | Pipeline Components | Tuning |
|---|---|---|
| **Logistic Regression** | RobustScaler → SMOTE → LogisticRegression | GridSearchCV (C, penalty, solver, class_weight) |
| **Random Forest** | SMOTE → RandomForestClassifier | Pre-tuned (300 trees, max_depth=10, balanced weights) |

### Evaluation Strategy

- **Cross-Validation:** 5-Fold Stratified K-Fold
- **Primary Metric:** F1-Score (balances precision and recall)
- **Additional Metrics:** Accuracy, Precision, Recall, ROC-AUC
- **Confusion Matrix:** Visualized with Seaborn heatmaps

### Why Logistic Regression Was Chosen for Production

Despite training both models, **Logistic Regression was selected as the production model** because:

1. **Interpretability** — Coefficients directly show feature impact direction and magnitude, essential for stakeholder trust.
2. **Probability Calibration** — Logistic Regression produces well-calibrated probability scores out-of-the-box, critical for the retention priority system.
3. **Speed** — Faster inference for real-time or batch scoring.
4. **Competitive Performance** — After hyperparameter tuning with SMOTE, it achieved strong F1 and ROC-AUC scores.

---

## 🛡️ Customer Retention System

After Part 4's model training, the notebook includes a **Customer Retention System** section that goes beyond prediction to create **actionable business intelligence**:

```
Churn Probability → CLV Calculation → Priority Score → Priority Level → Smart Retention Action
```

**Priority Score Formula:**
```
Priority Score = (Churn_Probability × 0.7 + Normalized_CLV × 0.3) × 100
```

This ensures that **high-value customers who are likely to churn** get the most aggressive retention efforts, while low-risk customers receive lighter-touch engagement.

The output is a **retention report** sorted by priority score, giving business teams a ready-to-action customer list.

---

## 💰 Customer Revenue Estimator

A separate **Customer Revenue Estimator** section quantifies the financial impact of churn:

- **Revenue At Risk** per customer = `MonthlyCharges × Churn_Probability`
- **Total Expected Monthly Revenue Loss** = Sum of all individual revenue-at-risk values
- **Top 10 At-Risk Customers** — Displayed with their monthly charges, churn probability, revenue at risk, and recommended retention action

This gives leadership a **dollar figure** to justify retention program budgets.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Pandas / NumPy** | Data manipulation and numerical operations |
| **Plotly** | Interactive visualizations (dark theme) |
| **Seaborn / Matplotlib** | Static visualizations (heatmaps, confusion matrices) |
| **Scikit-Learn** | ML pipelines, model training, evaluation, preprocessing |
| **Imbalanced-Learn (SMOTE)** | Handling class imbalance via synthetic oversampling |
| **Joblib** | Model serialization for production deployment |
| **KaggleHub** | Automated dataset download |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn imbalanced-learn kagglehub joblib
```

### Run the Notebook
```bash
jupyter notebook telco-customer-churn-prediction-1.ipynb
```

> **Note:** Part 1–3 use `kagglehub` to automatically download the dataset (requires Kaggle credentials). Part 4 reads from `processed_telco_churn.csv` which is included in this repo, so you can run Part 4 onward without Kaggle access.

---

## 📁 Project Structure

```
Customer-churn-prediction/
├── telco-customer-churn-prediction-1.ipynb   # Main notebook (Parts 1-4 + Retention + Revenue)
├── processed_telco_churn.csv                 # Cleaned & encoded dataset (output of Part 3, input to Part 4)
└── README.md                                 # You are here
```

---

## 🔑 Key Takeaways

1. **Month-to-month contracts** are the single biggest churn risk factor — lock customers into longer contracts with incentives.
2. **Customers without Online Security and Tech Support** churn at much higher rates — bundle these services.
3. **Fiber optic users** churn more, likely due to higher costs — consider competitive pricing for this segment.
4. **Electronic check payers** have the highest churn — encourage migration to auto-pay methods.
5. **New customers (low tenure)** are the most vulnerable — invest in strong onboarding programs.
6. **Higher monthly charges** correlate with churn — introduce loyalty pricing tiers for long-term customers.

---

## 🔮 Future Improvements

- [ ] Add **XGBoost and LightGBM** with full hyperparameter tuning for model comparison
- [ ] Implement **SHAP values** for model explainability at the individual prediction level
- [ ] Build a **Streamlit / Flask dashboard** for real-time churn prediction and retention monitoring
- [ ] Add **time-series analysis** to predict *when* a customer will churn, not just *if*
- [ ] Integrate with **CRM systems** for automated retention action triggering
- [ ] A/B test retention strategies to measure actual impact on churn reduction

---

<p align="center">
  <b>⭐ If you found this project useful, give it a star!</b>
</p>