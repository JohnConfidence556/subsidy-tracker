import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model, features = joblib.load("random_forest_model.pkl")

st.title("Subsidy Fraud Detection System")

# Create inputs for the required features
user_input = {
    "income_level": st.selectbox("Income Level", ["low", "medium", "high"]),
    "household_dependents": st.number_input("Number of Dependents", min_value=0, step=1),
    "monthly_energy_consumption_kwh": st.number_input("Energy Consumption (kWh)", min_value=0, step=1),
    "subsidy_eligibility": st.selectbox("Subsidy Eligibility", [0, 1]),
    "subsidy_type": st.selectbox("Subsidy Type", ["fuel", "food", "cash"]),
    "amount_(ngn)": st.number_input("Subsidy Amount (NGN)", min_value=0, step=100),
    "channel": st.selectbox("Channel", ["bank", "mobile_wallet", "cash"]),
    "wallet_activity_status": st.selectbox("Wallet Status", ["active", "inactive"]),
    "wallet_balance_(ngn)": st.number_input("Wallet Balance", min_value=0.0, step=100.0),
    "days_since_last_transaction": st.number_input("Days Since Last Transaction", min_value=0, step=1),
    "avg_monthly_wallet_balance": st.number_input("Avg Monthly Wallet Balance", min_value=0.0, step=100.0),
    "age": st.number_input("Age", min_value=18, step=1),
    "gender": st.selectbox("Gender", ["male", "female"]),
    "region": st.text_input("Region"),
    "age_group": st.selectbox("Age Group", ["18-30", "31-50", "51+"]),
    "isolation_forest_flag": st.selectbox("Isolation Forest Flag", [0, 1])
}

# Convert to DataFrame in correct column order
input_df = pd.DataFrame([[user_input[col] for col in features]], columns=features)

# Predict
if st.button("Predict Fraud"):
    prediction = model.predict(input_df)[0]
    st.write("Prediction:", "Fraudulent" if prediction == 1 else "Legit")
