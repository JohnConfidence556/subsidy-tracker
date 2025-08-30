%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

# Load pipeline
model = joblib.load("fraud_detection.pkl")

st.set_page_config(page_title="Fraud Detection System", layout="centered")

st.title("🔍 Subsidy Fraud Detection System")

with st.form("fraud_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    region = st.selectbox("Region", ["Lagos", "Kano", "Oyo", "Yobe", "Rivers", "Kaduna"])
    income_level = st.selectbox("Income Level", ["Low", "Middle", "High"])
    household_dependents = st.number_input("Household Dependents", min_value=0, value=2)
    monthly_energy_consumption_kwh = st.number_input("Monthly Energy Consumption (kWh)", min_value=0.0, value=450.5)
    subsidy_eligibility = st.selectbox("Subsidy Eligibility", [0, 1])
    subsidy_type = st.selectbox("Subsidy Type", ["Food", "Cash Transfer", "Fuel"])
    amount_ngn = st.number_input("Subsidy Amount (NGN)", min_value=0.0, value=25000.0)
    channel = st.selectbox("Channel", ["Bank Account", "Mobile Wallet", "Cash Pickup"])
    wallet_balance_ngn = st.number_input("Wallet Balance (NGN)", min_value=0.0, value=15000.0)
    days_since_last_transaction = st.number_input("Days Since Last Transaction", min_value=0, value=5)
    avg_monthly_wallet_balance = st.number_input("Average Monthly Wallet Balance", min_value=0.0, value=12000.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "region": [region],
        "income_level": [income_level],
        "household_dependents": [household_dependents],
        "monthly_energy_consumption_kwh": [monthly_energy_consumption_kwh],
        "subsidy_eligibility": [subsidy_eligibility],
        "subsidy_type": [subsidy_type],
        "amount_(ngn)": [amount_ngn],
        "channel": [channel],
        "wallet_balance_(ngn)": [wallet_balance_ngn],
        "days_since_last_transaction": [days_since_last_transaction],
        "avg_monthly_wallet_balance": [avg_monthly_wallet_balance]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ Not Fraud")
    else:
        st.error("⚠️ Fraud Suspected")

    # Show result
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of fraud: {prob:.2f})")
