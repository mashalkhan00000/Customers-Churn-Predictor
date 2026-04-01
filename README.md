# Customer Churn Predictor

A machine learning project that predicts whether a telecom customer is likely to churn (cancel their service) — built with Python, Scikit-learn, and Streamlit.

---

## What This Project Does

Given a customer's profile (contract type, tenure, monthly charges, services used etc.), the model predicts:
- Whether the customer will **churn or stay**
- The **probability** of churn with a risk level indicator
- Which **features drive churn** the most

---

## Project Structure

```
customer_churn_predictor/
│
├── data/
│   └── telco_churn.csv          # Dataset 
│
├── models/                      # Auto-created after training
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── feature_names.pkl
│
├── outputs/                     # Auto-created after training
│   ├── churn_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
├── data_preprocessing.py        # Data cleaning, encoding, scaling
├── model_training.py            # Train & compare multiple ML models
├── visualizations.py            # All charts and plots
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Dataset

Download the **Telco Customer Churn** dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV file inside the `data/` folder and rename it to `telco_churn.csv`.

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python model_training.py
```
This will:
- Clean and preprocess the data
- Train 4 ML models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Compare performance and save the best one
- Generate all charts into the `/outputs` folder

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## Models Used

| Model | Notes |
|---|---|
| Logistic Regression | Fast, interpretable baseline |
| Random Forest | Strong ensemble, handles non-linearity |
| Gradient Boosting | Usually best accuracy |
| SVM | Good for smaller datasets |

The best model is selected automatically based on ROC-AUC score.

---

## Tech Stack

- **Python 3.8+**
- **Pandas & NumPy** — data manipulation
- **Scikit-learn** — ML models & preprocessing
- **Matplotlib & Seaborn** — visualizations
- **Streamlit** — interactive web dashboard
- **Joblib** — model serialization

---
## Results

Typical performance on Telco Churn dataset:
- Accuracy: ~80–82%
- ROC-AUC: ~84–86%

---

*Built as a portfolio project for Fiverr — Machine Learning & Data Science services.*
