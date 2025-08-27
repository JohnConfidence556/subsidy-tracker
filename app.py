import streamlit as st
import pandas as pd
import joblib

# =========================
# Load trained pipeline
# =========================
pipeline = joblib.load("fraud_detection_pipeline.pkl")
model = pipeline["model"]
scaler = pipeline["scaler"]
encoders = pipeline["encoders"]
features = pipeline["features"]

st.title("💳 Subsidy Fraud Detection App")
st.write("Enter applicant details to check the probability of subsidy fraud.")

# =========================
# Collect user input
# =========================
user_data = {
    "gender": st.selectbox("Gender", ["male", "female"]),
    "region": st.selectbox("Region", ["north", "south", "east", "west"]),
    "income_level": st.selectbox("Income Level", ["low", "medium", "high"]),
    "subsidy_type": st.selectbox("Subsidy Type", ["fuel", "food", "cash"]),
    "channel": st.selectbox("Channel", ["mobile", "bank", "agent"]),
    "wallet_activity_status": st.selectbox("Wallet Activity Status", ["active", "inactive"]),
    "year_month": st.text_input("Year-Month (e.g. 2025-01)", "2025-01"),
    "amount_(ngn)": st.number_input("Transaction Amount (NGN)", min_value=0.0, step=100.0),
    "wallet_balance_(ngn)": st.number_input("Wallet Balance (NGN)", min_value=0.0, step=100.0),
    "avg_monthly_wallet_balance": st.number_input("Avg Monthly Wallet Balance (NGN)", min_value=0.0, step=100.0),
    "days_since_last_transaction": st.number_input("Days Since Last Transaction", min_value=0, step=1),
    "subsidy_eligibility": st.selectbox("Subsidy Eligibility", [0, 1]),
    "isolation_forest_flag": st.selectbox("Isolation Forest Flag", [0, 1])
}

# =========================
# Preprocess input
# =========================
def preprocess_user_input(user_data: dict):
    df = pd.DataFrame([user_data])

    # Encode categorical columns with stored encoders
    for col, le in encoders.items():
        df[col] = le.transform(df[col].astype(str))

    # Scale numeric + categorical in correct order
    df_scaled = scaler.transform(df[features])
    return df_scaled

# =========================
# Make Prediction
# =========================
if st.button("Predict Fraud Risk"):
    input_scaled = preprocess_user_input(user_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Potential Fraud Detected! (Risk Score: {proba:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Risk Score: {proba:.2f})")
