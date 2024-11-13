import streamlit as st
import joblib
import pandas as pd
import datetime
import requests

# Load the trained model and scaler
Model = joblib.load("loan_approval_model.pkl")
Scaler = joblib.load("rb_scaler.pkl")

# Function to fetch interest rate and inflation rate from Trading Economics API for India
# Free Accounts have only access to limited countries so manually updated.

# Streamlit UI for loan prediction
st.title("Loan Approval Prediction")
st.header("Enter Loan Details")

# User inputs
dependents = int(st.text_input("Number of dependents", "0"))
education = int(st.selectbox("Education", [0, 1]))  # 0: Not Graduate, 1: Graduate
self_employed = int(st.selectbox("Self Employed", [0, 1]))  # 0: No, 1: Yes
annual_income = float(st.text_input("Annual Income", "200000"))
loan_amount = float(st.text_input("Loan Amount", "300000"))
loan_term = int(st.text_input("Loan Term (years)", "2"))
credit_score = float(st.text_input("Credit Score", "300"))
residential_av = float(st.text_input("Residential AV", "0"))
commercial_av = float(st.text_input("Commercial AV", "0"))
luxury_av = float(st.text_input("Luxury AV", "0"))
bank_av = float(st.text_input("Bank AV", "0"))

# Determine the current season based on the month
loan_date = datetime.datetime.now()
month = loan_date.month

if month in [3, 4, 5]:
    season = 1  # Spring
elif month in [6, 7, 8]:
    season = 2  # Summer
elif month in [9, 10, 11]:
    season = 3  # Fall
else:
    season = 0  # Winter

# Fetch interest rate and inflation rate from Trading Economics API for India

interest_rate = 6.50
inflation_rate = 5.49

st.write(f"Current Interest Rate: {interest_rate}")
st.write(f"Current Inflation Rate: {inflation_rate}")
# Prepare input data for model prediction
input_data = pd.DataFrame({
    'dependents': [dependents],
    'education': [education],
    'self_employed': [self_employed],
    'annual_income': [annual_income],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'credit_score': [credit_score],
    'residential_av': [residential_av],
    'commercial_av': [commercial_av],
    'luxury_av': [luxury_av],
    'bank_av': [bank_av],
    'interest_rate': [interest_rate],
    'inflation_rate': [inflation_rate],
    'season': [season]  # Keep season here, but it will not be scaled
})

# Scale numerical columns excluding 'season'
numerical_cols = ['annual_income', 'loan_amount', 'loan_term', 'credit_score',
                  'residential_av', 'commercial_av', 'luxury_av', 'bank_av',
                  'interest_rate', 'inflation_rate']  # Removed 'season' from scaling
temp_df = pd.DataFrame(Scaler.transform(input_data[numerical_cols]), columns=numerical_cols)
input_data.drop(numerical_cols, axis=1, inplace=True)
input_data = pd.concat([input_data[['dependents', 'education', 'self_employed', 'season']], temp_df], axis=1)

# Predict loan approval
if st.button("Predict"):
    prediction_probs = Model.predict_proba(input_data)  # Get prediction probabilities
    result = int(prediction_probs[0][1] >= 0.5)  # Convert to binary: 1 if probability >= 0.5, else 0
    label = ["rejected", "approved"]
    st.markdown(f"## The application is **{label[result]}**.")
