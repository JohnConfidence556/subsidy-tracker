import streamlit as st
import pandas as pd
import joblib

# =========================
# Load the trained model
# =========================
model, features = joblib.load("random_forest_model.pkl")  

st.title("Subsidy Fraud Detection System 🚨")

st.write("""
This app predicts whether a subsidy transaction is **suspicious or not**  
based on demographic, financial, and transaction information.
""")

# =========================
# Collect User Input
# =========================
st.header("Enter Beneficiary / Transaction Details")

with st.form("fraud_form"):
    income_level = st.selectbox("Income Level", ["low", "medium", "high"])
    subsidy_type = st.selectbox("Subsidy Type", ["fuel", "food", "cash"])
    amount = st.number_input("Subsidy Amount (NGN)", min_value=0, step=100)
    channel = st.selectbox("Disbursement Channel", ["bank", "mobile_wallet", "cash"])
    wallet_activity_status = st.selectbox("Wallet Activity Status", ["active", "inactive"])
    wallet_balance = st.number_input("Wallet Balance (NGN)", min_value=0.0, step=100.0)
    avg_monthly_wallet_balance = st.number_input("Average Monthly Wallet Balance (NGN)", min_value=0.0, step=100.0)
    days_since_last_transaction = st.number_input("Days Since Last Transaction", min_value=0, step=1)
    isolation_forest_flag = st.selectbox("Isolation Forest Flag (anomaly)", [0, 1])  # Quick manual option

    submitted = st.form_submit_button("Predict Fraud")

# =========================
# Make Prediction
# =========================
if submitted:
    # Create input dictionary
    user_data = {
        "income_level": income_level,
        "subsidy_type": subsidy_type,
        "amount_(ngn)": amount,
        "channel": channel,
        "wallet_activity_status": wallet_activity_status,
        "wallet_balance_(ngn)": wallet_balance,
        "avg_monthly_wallet_balance": avg_monthly_wallet_balance,
        "days_since_last_transaction": days_since_last_transaction,
        "isolation_forest_flag": isolation_forest_flag,
    }

    # Convert to DataFrame with correct feature order
    input_df = pd.DataFrame([user_data], columns=features)

    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # probability of fraud

    # =========================
    # Show Results
    # =========================
    if prediction == 1:
        st.error(f"🚨 Suspicious Transaction Detected! (Fraud Probability: {proba:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {proba:.2%})")


    if prediction == 1:
        st.error("⚠️ Suspicious transaction detected (Possible Fraud)")
    else:
        st.success("✅ Transaction looks normal")
