import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("fraud_detection_pipeline.pkl")
model = pipeline["model"]
scaler = pipeline["scaler"]
encoders = pipeline["encoders"]
features = pipeline["features"]

st.title("Subsidy Fraud Detection App")

st.header("Enter Transaction Details")

# --- Collect All 15 Features ---
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", encoders["gender"].classes_.tolist())
region = st.selectbox("Region", encoders["region"].classes_.tolist())
income_level = st.selectbox("Income Level", encoders["income_level"].classes_.tolist())
household_dependents = st.number_input("Household Dependents", min_value=0, step=1)
monthly_energy_consumption_kwh = st.number_input("Monthly Energy Consumption (kWh)", min_value=0, step=10)
subsidy_type = st.selectbox("Subsidy Type", encoders["subsidy_type"].classes_.tolist())
amount_ngn = st.number_input("Subsidy Amount (NGN)", min_value=0, step=100)
channel = st.selectbox("Disbursement Channel", encoders["channel"].classes_.tolist())
wallet_activity_status = st.selectbox("Wallet Activity Status", encoders["wallet_activity_status"].classes_.tolist())
wallet_balance_ngn = st.number_input("Wallet Balance (NGN)", min_value=0, step=100)
days_since_last_transaction = st.number_input("Days Since Last Transaction", min_value=0, step=1)
avg_monthly_wallet_balance = st.number_input("Average Monthly Wallet Balance (NGN)", min_value=0, step=100)
isolation_forest_flag = st.selectbox("Isolation Forest Flag", [0, 1])

# --- Build input DataFrame ---
user_data = {
    "age": age,
    "gender": gender,
    "region": region,
    "income_level": income_level,
    "household_dependents": household_dependents,
    "monthly_energy_consumption_kwh": monthly_energy_consumption_kwh,
    "subsidy_type": subsidy_type,
    "amount_(ngn)": amount_ngn,
    "channel": channel,
    "wallet_activity_status": wallet_activity_status,
    "wallet_balance_(ngn)": wallet_balance_ngn,
    "days_since_last_transaction": days_since_last_transaction,
    "avg_monthly_wallet_balance": avg_monthly_wallet_balance,
    "isolation_forest_flag": isolation_forest_flag
}
input_df = pd.DataFrame([user_data])

# --- Encode categorical features ---
for col, encoder in encoders.items():
    if col in input_df.columns:
        input_df[col] = encoder.transform(input_df[col])

# --- Scale numeric features ---
numeric_cols = [
    "age",
    "household_dependents",
    "monthly_energy_consumption_kwh",
    "amount_(ngn)",
    "wallet_balance_(ngn)",
    "days_since_last_transaction",
    "avg_monthly_wallet_balance",
    "isolation_forest_flag"
]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# --- Align features to training ---
input_df = input_df.reindex(columns=features, fill_value=0)

# --- Predict ---
if st.button("Predict Fraud"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Transaction is likely FRAUDULENT (probability: {proba:.2%})")
    else:
        st.success(f"✅ Transaction looks NORMAL (fraud probability: {proba:.2%})")

