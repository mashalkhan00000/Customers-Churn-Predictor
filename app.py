import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

@st.cache_resource
def load_model():
    model         = joblib.load("models/best_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    encoders      = joblib.load("models/encoders.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, encoders, feature_names

model, scaler, encoders, feature_names = load_model()

st.title("📉 Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to leave — before it's too late.")
st.divider()

st.sidebar.header("Customer Details")
st.sidebar.markdown("Fill in the customer profile below:")

def user_input():
    tenure          = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 10, 120, 65)
    total_charges   = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
    contract        = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet        = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment         = st.sidebar.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])
    senior          = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    dependents      = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
    tech_support    = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])

    data = {
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "Contract":         contract,
        "InternetService":  internet,
        "PaymentMethod":    payment,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          partner,
        "Dependents":       dependents,
        "TechSupport":      tech_support,
        "OnlineSecurity":   online_security,
    }
    return pd.DataFrame([data])

input_df = user_input()

def preprocess_input(df):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return scaler.transform(df)

processed   = preprocess_input(input_df)
prediction  = model.predict(processed)[0]
probability = model.predict_proba(processed)[0]
churn_prob  = probability[1]
stay_prob   = probability[0]

# Prediction result
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Churn Probability", f"{churn_prob:.1%}")
with col2:
    st.metric("Retention Probability", f"{stay_prob:.1%}")
with col3:
    result = "⚠️ Likely to Churn" if prediction == 1 else "✅ Likely to Stay"
    st.metric("Prediction", result)

st.divider()

# Risk gauge
st.subheader("Risk Level")
if churn_prob < 0.35:
    st.success(f"Low Risk — This customer is {stay_prob:.0%} likely to stay.")
elif churn_prob < 0.65:
    st.warning(f"Medium Risk — Monitor this customer. Churn probability: {churn_prob:.0%}")
else:
    st.error(f"High Risk — Immediate action recommended! Churn probability: {churn_prob:.0%}")

st.progress(float(churn_prob))
st.caption(f"Churn probability: {churn_prob:.1%}")

st.divider()

# Customer profile table
st.subheader("Customer Profile")
st.dataframe(input_df.T.rename(columns={0: "Value"}).astype(str), use_container_width=True)

st.divider()

# Model performance charts
st.subheader("Model Performance Overview")
chart_files = {
    "Model Comparison":   "outputs/model_comparison.png",
    "ROC Curves":         "outputs/roc_curves.png",
    "Confusion Matrix":   "outputs/confusion_matrix.png",
    "Feature Importance": "outputs/feature_importance.png",
}

charts = [(t, p) for t, p in chart_files.items() if os.path.exists(p)]
for title, path in charts:
    st.subheader(title)
    st.image(path, use_container_width=True)
    st.divider()

st.caption("Built with Python · Scikit-learn · Streamlit — Customer Churn Predictor")
