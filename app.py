import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model + encoders + scaler
# =========================
pipeline = joblib.load("fraud_detection_pipeline.pkl")
model = pipeline["model"]
scaler = pipeline["scaler"]
encoders = pipeline["encoders"]
features = pipeline["features"]

st.title("Subsidy Fraud Detection System 🚨")

st.write("This app predicts whether a subsidy transaction is **suspicious or not**")

# =========================
# Collect User Input
# =========================
with st.form("fraud_form"):
    gender = st.selectbox("Gender", ["male", "female"])
    region = st.selectbox("Region", ["north", "south", "east", "west"])  # replace with actual regions
    income_level = st.selectbox("Income Level", ["low", "medium", "high"])
    subsidy_type = st.selectbox("Subsidy Type", ["fuel", "food", "cash"])
    channel = st.selectbox("Disbursement Channel", ["bank", "mobile_wallet", "cash"])
    wallet_activity_status = st.selectbox("Wallet Activity Status", ["active", "inactive"])
    year_month = st.text_input("Year-Month (e.g., 2025-08)")
    amount = st.number_input("Subsidy Amount (NGN)", min_value=0, step=100)
    wallet_balance = st.number_input("Wallet Balance (NGN)", min_value=0.0, step=100.0)
    avg_monthly_wallet_balance = st.number_input("Average Monthly Wallet Balance (NGN)", min_value=0.0, step=100.0)
    days_since_last_transaction = st.number_input("Days Since Last Transaction", min_value=0, step=1)
    isolation_forest_flag = st.selectbox("Isolation Forest Flag", [0, 1])

    submitted = st.form_submit_button("Predict Fraud")

# =========================
# Make Prediction
# =========================
if submitted:
    # Prepare data as DataFrame
    user_data = {
        "gender": gender,
        "region": region,
        "income_level": income_level,
        "subsidy_type": subsidy_type,
        "channel": channel,
        "wallet_activity_status": wallet_activity_status,
        "year_month": year_month,
        "amount_(ngn)": amount,
        "wallet_balance_(ngn)": wallet_balance,
        "avg_monthly_wallet_balance": avg_monthly_wallet_balance,
        "days_since_last_transaction": days_since_last_transaction,
        "isolation_forest_flag": isolation_forest_flag
    }
    df = pd.DataFrame([user_data])

    # Apply encoders
    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    # Scale numerical values
    df_scaled = scaler.transform(df[features])

    # Predict
    prediction = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"🚨 Suspicious Transaction Detected! (Fraud Probability: {proba:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {proba:.2%})")
