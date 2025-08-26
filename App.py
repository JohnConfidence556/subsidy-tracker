import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

st.title("Subsidy Fraud Detection System")

# Collect inputs from user
age = st.number_input("Age", min_value=18, max_value=100, step=1)
income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
wallet_activity_status = st.selectbox("Wallet Activity", ["Active", "Inactive"])
channel = st.selectbox("Disbursement Channel", ["Bank Transfer", "Mobile Money", "Agent"])
amount = st.number_input("Subsidy Amount (NGN)", min_value=0, step=100)
wallet_balance = st.number_input("Wallet Balance (NGN)", min_value=0, step=100)
avg_balance = st.number_input("Average Monthly Wallet Balance (NGN)", min_value=0, step=100)
days_inactive = st.number_input("Days Since Last Transaction", min_value=0, step=1)

# Convert categorical values to numerical (same preprocessing as training)
# Example: encode manually or use same encoder you saved during training
income_map = {"Low": 0, "Medium": 1, "High": 2}
wallet_map = {"Inactive": 0, "Active": 1}
channel_map = {"Bank Transfer": 0, "Mobile Money": 1, "Agent": 2}

# Feature vector
features = [[
    age,
    income_map[income_level],
    wallet_map[wallet_activity_status],
    channel_map[channel],
    amount,
    wallet_balance,
    avg_balance,
    days_inactive
]]

# Prediction
if st.button("Check Fraud Risk"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("🚨 Suspicious Transaction Detected! (Fraud Risk)")
    else:
        st.success("✅ Transaction Looks Normal")