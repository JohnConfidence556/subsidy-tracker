import streamlit as st
import pandas as pd
import joblib

# =========================
# Load saved pipeline
# =========================
pipeline = joblib.load("fraud_detection_pipeline.pkl")

model = pipeline["model"]
scaler = pipeline["scaler"]
encoders = pipeline["encoders"]
features = pipeline["features"]

st.title("🕵️ Subsidy Fraud Detection System")

# =========================
# User input form
# =========================
user_data = {}

# Example categorical inputs
user_data["gender"] = st.selectbox("Gender", ["male", "female"])
user_data["region"] = st.selectbox("Region", ["north", "south", "east", "west"])
user_data["income_level"] = st.selectbox("Income Level", ["low", "medium", "high"])
user_data["subsidy_type"] = st.selectbox("Subsidy Type", ["fuel", "food", "cash"])
user_data["channel"] = st.selectbox("Channel", ["mobile", "bank", "agent"])
user_data["wallet_activity_status"] = st.selectbox("Wallet Activity", ["active", "inactive"])
user_data["year_month"] = st.text_input("Year-Month (e.g. 2022-01)", "2022-01")

# Example numeric inputs
user_data["amount_(ngn)"] = st.number_input("Amount (NGN)", 0, 100000, 5000)
user_data["wallet_balance_(ngn)"] = st.number_input("Wallet Balance (NGN)", 0, 200000, 10000)
user_data["avg_monthly_wallet_balance"] = st.number_input("Average Monthly Wallet Balance", 0, 200000, 12000)
user_data["days_since_last_transaction"] = st.number_input("Days Since Last Transaction", 0, 365, 30)

# Add anomaly flag manually if you want users to set it (optional)
user_data["isolation_forest_flag"] = st.selectbox("Isolation Forest Flag", [0, 1])

# =========================
# Prediction
# =========================
if st.button("Predict Fraud"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_data])

    # Apply encoders for categorical columns
    for col, le in encoders.items():
        if col in input_df:
            input_df[col] = le.transform(input_df[col].astype(str))

    # Apply scaling
    input_scaled = scaler.transform(input_df[features])

    # Predict
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ Fraudulent transaction detected with probability {proba:.2f}")
    else:
        st.success(f"✅ Legitimate transaction with probability {1 - proba:.2f}")
