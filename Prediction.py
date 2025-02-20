# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:19:28 2025

@author: Owner
"""

import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("ml_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Healthcare Cost Prediction App")
st.write("Enter healthcare factors to predict total costs.")

# Define input fields based on the trained model
year = st.number_input("Year", min_value=2007, max_value=2014, step=1)
county = st.number_input("County (Numeric Code)", min_value=0, max_value=39, step=1)
fips_code = st.number_input("State and County FIPS Code (Numeric)", min_value=53001, max_value=53075, step=1)
standardized_costs = st.number_input("Total Standardized Costs ($)", min_value=0.0, step=1000.0)
risk_adjusted_costs = st.number_input("Total Standardized Risk-Adjusted Costs ($)", min_value=0.0, step=1000.0)
per_capita_costs = st.number_input("Actual Per Capita Costs ($)", min_value=0.0, step=100.0)
standardized_per_capita_costs = st.number_input("Standardized Per Capita Costs ($)", min_value=0.0, step=100.0)
standardized_risk_adjusted_per_capita_costs = st.number_input("Standardized Risk-Adjusted Per Capita Costs ($)", min_value=0.0, step=100.0)
readmission_rate = st.number_input("Hospital Readmission Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
emergency_visits = st.number_input("Emergency Department Visits", min_value=0, step=1)

# Predict function
if st.button("Predict Total Costs"):
    # Create a feature array
    user_data = np.array([[year, county, fips_code, standardized_costs, risk_adjusted_costs, 
                           per_capita_costs, standardized_per_capita_costs,
                           standardized_risk_adjusted_per_capita_costs, readmission_rate, emergency_visits]])

    # Debugging: Check input shape before transformation
    st.write(f"🛠️ Debugging: Input feature shape: {user_data.shape}")
    
    # Ensure it matches the trained model
    try:
        user_data_scaled = scaler.transform(user_data)  # Apply same transformation as during training
        prediction = model.predict(user_data_scaled)[0]
        st.success(f"Predicted Total Actual Costs: ${prediction:,.2f}")
    except ValueError as e:
        st.error(f"❌ Error: Feature Mismatch - {str(e)}")
        st.write(f"Expected {scaler.n_features_in_} features, but got {user_data.shape[1]}")

