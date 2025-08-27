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

# Add Gender back
gender = st.selectbox("Gender", encoders["gender"].classes_.tolist())
income_level = st.selectbox("Income Level", encoders["income_level"].classes_.tolist())
subsidy_type = st.selectbox("Subsidy Type", encoders["subsidy_type"].classes_.tolist())
channel = st.selectbox("Disbursement Channel", encoders["channel"].classes_.tolist())
wallet_activity_status = st.selectbox("Wallet Activity Status", encoders["wallet_activity_status"].classes_.tolist())

amount = st.number_input("Subsidy Amount (NGN)", min_value=0, step=100)
wallet_balance = st.number_input("Wallet Balance (NGN)", min_value=0.0, step=100.0)
avg_monthly_wallet_balance = st.number_input("Average Monthly Wallet Balance (NGN)", min_value=0.0, step=100.0)
days_since_last_transaction = st.number_input("Days Since Last Transaction", min_value=0, step=1)
isolation_forest_flag = st.selectbox("Isolation Forest Flag", [0, 1])

# Build input DataFrame
user_data = {
    "gender": gender,
    "income_level": income_level,
    "subsidy_type": subsidy_type,
    "channel": channel,
    "wallet_activity_status": wallet_activity_status,
    "amount": amount,
    "wallet_balance": wallet_balance,
    "avg_monthly_wallet_balance": avg_monthly_wallet_balance,
    "days_since_last_transaction": days_since_last_transaction,
    "isolation_forest_flag": isolation_forest_flag
}
input_df = pd.DataFrame([user_data])

# Encode categorical
for col, encoder in encoders.items():
    if col in input_df.columns:
        input_df[col] = encoder.transform(input_df[col])

# Scale numeric
numeric_cols = ["amount", "wallet_balance", "avg_monthly_wallet_balance",
                "days_since_last_transaction", "isolation_forest_flag"]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Align features
input_df = input_df.reindex(columns=features, fill_value=0)

# Predict
if st.button("Predict Fraud"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Transaction is likely FRAUDULENT (probability: {proba:.2%})")
    else:
        st.success(f"✅ Transaction looks NORMAL (fraud probability: {proba:.2%})")
