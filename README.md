# 🏦 LoanSense - Intelligent Loan Approval System

> An intelligent machine learning system built for **SecureTrust Bank** to automate and improve loan approval decisions using historical applicant data.

[![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat&logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat)](https://xgboost.readthedocs.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.3%25-brightgreen?style=flat)]()
[![F1 Score](https://img.shields.io/badge/F1%20Score-0.94-brightgreen?style=flat)]()

---

## 📌 Problem Statement

SecureTrust Bank processes hundreds of loan applications daily across urban and rural regions of India. Their manual verification process — checking income proofs, employment details, and credit history — is slow, inconsistent, and prone to human bias.

This leads to two costly outcomes:
- **Good customers get rejected** → loss of business
- **High-risk customers get approved** → financial losses

**CreditWise** solves this by predicting whether a loan should be approved or rejected before final human verification, using patterns learned from historical loan data.

---

## 🎯 Objective

Build a supervised machine learning classification system that:
- Analyses 20 financial and personal features of a loan applicant
- Predicts loan approval (`1 = Approved`, `0 = Rejected`)
- Provides a confidence score with each decision
- Deploys as an interactive web application for bank staff

---

## 🗂 Dataset Description

| Property | Detail |
|---|---|
| Source | SecureTrust Bank historical loan records |
| Total Records | 1,000 applicants |
| After Cleaning | 950 applicants |
| Features | 20 |
| Target | `Loan_Approved` (1 = Approved, 0 = Rejected) |
| Class Distribution | 69% Rejected / 31% Approved |

**Key Features:**

| Feature | Description |
|---|---|
| `Applicant_Income` | Monthly income of applicant |
| `Coapplicant_Income` | Monthly income of co-applicant |
| `Credit_Score` | Credit bureau score (300–900) |
| `DTI_Ratio` | Debt-to-income ratio |
| `Loan_Amount` | Loan amount requested |
| `Loan_Term` | Loan duration in months |
| `Employment_Status` | Salaried / Self-Employed / Business |
| `Education_Level` | Graduate / Postgraduate / Undergraduate |
| `Property_Area` | Urban / Semi-Urban / Rural |
| `Collateral_Value` | Value of collateral provided |

---

## 🔬 Methodology

Raw Data → Drop Missing Targets → Handle Missing Values
→ Label Encoding + One-Hot Encoding → Correlation Analysis
→ Train-Test Split (Stratified 80/20) → Standard Scaling
→ Train 7 Models → Cross Validation → Final Evaluation
→ Select Best Model → Save → Deploy

**Preprocessing steps:**
- Dropped 50 rows with missing target labels
- Mean imputation for numerical features
- Most-frequent imputation for categorical features (excluding target)
- Label encoding for `Education_Level` (ordinal)
- One-hot encoding for remaining categorical features (`drop='first'`)
- StandardScaler fitted on training data only — applied to test via `transform()` only

---

## 🤖 Models Trained

Seven supervised classification models were trained and evaluated:

| Rank | Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| 🥇 | **XGBoost** | **0.9632** | **0.9206** | **0.9667** | **0.9431** |
| 🥈 | Random Forest | 0.9474 | 0.9032 | 0.9333 | 0.9180 |
| 🥉 | Decision Tree | 0.9421 | 0.9121 | 0.9034 | 0.9073 |
| 4 | SVM | 0.8645 | 0.8232 | 0.7273 | 0.7707 |
| 5 | Logistic Regression | 0.8618 | 0.7964 | 0.7520 | 0.7735 |
| 6 | Naive Bayes | 0.8684 | 0.8495 | 0.7063 | 0.7692 |
| 7 | KNN | 0.7566 | 0.6367 | 0.5253 | 0.5745 |

> **Primary metric: F1 Score** — chosen because both false approvals (financial loss) and false rejections (business loss) are costly for the bank.

---

## 🏆 Best Model — XGBoost

**Confusion Matrix on Test Set (190 applicants):**
Predicted NO    Predicted YES
Actual NO   →  [    125    ]    [     5     ]
Actual YES  →  [      2    ]    [    58     ]

- ✅ 125 risky applicants correctly rejected
- ✅ 58 good applicants correctly approved
- ❌ 5 risky applicants wrongly approved
- ❌ 2 good applicants wrongly rejected
- **Only 7 total mistakes out of 190 test applicants**

**Why XGBoost?**
XGBoost builds trees sequentially — each tree corrects the mistakes of the previous one. This boosting strategy makes it highly effective at catching difficult edge cases, which is exactly why its recall (0.9667) is the highest of all models. It identifies almost every genuinely creditworthy applicant.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Ensemble / Boosting | XGBoost |
| Model Saving | Joblib |
| Web Application | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure
credit-loan-prediction/
│
├── credit-loan-prediction.ipynb   # Full ML pipeline notebook
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
│
├── xgb_model.pkl                  # Trained XGBoost model
├── scaler.pkl                     # Fitted StandardScaler
├── ohe.pkl                        # Fitted OneHotEncoder
├── le_edu.pkl                     # Fitted LabelEncoder (Education_Level)
│
├── loan_approval_data             # Dataset (CSV)
├── .gitignore
└── README.md

---

## 🚀 Run Locally

**1. Clone the repository**
git clone [https://github.com/vishalvermacore/LoanSense.git](https://github.com/vishalvermacore/LoanSense)
cd credit-loan-prediction

**2. Install dependencies**
pip install -r requirements.txt

**3. Launch the app**
streamlit run app.py

App opens at `http://localhost:8501`

---

## ☁️ Deployment

The app is deployed on **Streamlit Community Cloud**.

🔗 **Live App:** [credit-loan-prediction.app](https://credit-loan-prediction.streamlit.app/)

---

## 📊 Key Findings

- **Credit Score and DTI Ratio** were the most influential features in loan approval decisions
- **XGBoost outperformed all other models** across every metric — accuracy, precision, recall, and F1
- **KNN performed the worst** (F1: 0.57) and is not suitable for this use case
- **Class imbalance (69/31)** was handled via stratified train-test splitting and F1 as the primary metric
- **Ensemble models (XGBoost, Random Forest)** significantly outperformed single models, confirming that boosting and bagging are better suited for structured financial data

---

## 👤 Author

**Vishal Verma**
Muzaffarnagar, Uttar Pradesh, India

---

## ⚠️ Disclaimer

This system is built for academic and internal bank demonstration purposes. Final loan decisions should always involve human verification and comply with applicable financial regulations.
