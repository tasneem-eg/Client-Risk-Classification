# 🏦 Credit Risk Classification System

A complete Machine Learning pipeline for detecting high-risk banking clients using classification models, imbalance handling techniques, and an interactive Streamlit dashboard.

---

# 📌 Project Overview

This project predicts whether a client is likely to become a **BAD** or **GOOD** credit customer.

The system includes:

- Data preprocessing
- Feature engineering
- Handling missing values
- Imbalanced dataset treatment
- Training multiple ML models
- Model evaluation
- Threshold tuning
- Interactive Streamlit scoring dashboard

---

# 🎯 Business Problem

Bank datasets are highly imbalanced.

Most customers are GOOD clients, while only a very small percentage are BAD clients.

A normal model can achieve very high accuracy while completely failing to detect risky customers.

This project focuses on:

✅ Maximizing Recall  
✅ Catching as many BAD customers as possible  
✅ Maintaining strong AUC performance

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Missing value imputation
- Duplicate removal
- Label encoding
- Feature scaling
- Feature engineering

---

# 🤖 Machine Learning Models

The project compares multiple models:

- Logistic Regression
- Random Forest
- Decision Tree
- SVM

With different imbalance strategies:

- Class Weight Balancing
- SMOTE
- SMOTETomek
- SMOTEENN

Total trained models: **16**

---

# 🏆 Best Model

Best model selected:

## Logistic Regression (Balanced)

### Final Metrics

| Metric | Score |
|---|---|
| Accuracy | 0.732 |
| Recall | 0.756 |
| AUC | 0.796 |
| F1 Score | 0.104 |

---

# 📊 Dashboard Features

The Streamlit dashboard supports:

- Dataset upload
- Preprocessing visualization
- Model evaluation
- ROC Curve
- Confusion Matrix
- Single client scoring
- Batch client scoring

---

# 🖥️ Streamlit App

Run locally:

```bash
streamlit run App.py