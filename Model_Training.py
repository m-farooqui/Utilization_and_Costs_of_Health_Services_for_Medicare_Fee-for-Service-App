# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:18:47 2025

@author: Owner
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import re

# Load the dataset
file_path = "C:/Users/Owner/OneDrive/Documents/Datasets/Utilization_and_Costs_of_Health_Services_for_Medicare_Fee-for-Service_Beneficiaries__Washington_State_and_Counties__2007-2014.csv"
df = pd.read_csv(file_path)

# -------------------------------------
# 🛠️ **Step 1: Data Cleaning**
# -------------------------------------

# Convert column names to a standard format (remove extra spaces)
df.columns = df.columns.str.strip()

# Function to convert dollar values to float
def clean_dollar_values(value):
    if isinstance(value, str):
        return float(value.replace("$", "").replace(",", ""))
    return value

# Function to convert percentage values to float
def clean_percentage_values(value):
    if isinstance(value, str):
        return float(value.replace("%", "")) / 100
    return value

# Apply cleaning functions to relevant columns
for col in df.columns:
    if df[col].dtype == "object":  
        if df[col].str.contains(r"^\$[\d,]+\.?\d*$", na=False).any():  # Detects columns with "$"
            df[col] = df[col].apply(clean_dollar_values)
        elif df[col].str.contains(r"%$", na=False).any():  # Detects percentage columns
            df[col] = df[col].apply(clean_percentage_values)

# Convert numeric columns that should be integers
df["Year"] = df["Year"].astype(int)
df["State and County FIPS Code"] = df["State and County FIPS Code"].astype(str)  # Keep as string

# Convert categorical column `County` to numeric codes
df["County"] = df["County"].astype("category").cat.codes  

# -------------------------------------
# 🏗️ **Step 2: Feature Selection**
# -------------------------------------

# Define target variable
target_column = "Total Actual Costs"

# Selecting relevant features for prediction
features = [
    "Year", "County", "State and County FIPS Code", 
    "Total Standardized Costs", "Total Standardized Risk-Adjusted Costs",
    "Actual Per Capita Costs", "Standardized Per Capita Costs",
    "Standardized Risk-Adjusted Per Capita Costs", "Hospital Readmission Rate",
    "Emergency Department Visits"
]

X = df[features]
y = df[target_column]

# Convert FIPS code (categorical) to numeric encoding
X["State and County FIPS Code"] = X["State and County FIPS Code"].astype("category").cat.codes  

# -------------------------------------
# 🚀 **Step 3: Train ML Model**
# -------------------------------------

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "C:/Users/Owner/OneDrive/Documents/Utilization_and_Costs_of_Health_Services_for_Medicare_Fee-for-Service App/scaler.pkl")

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model Performance: MAE={mae:.2f}, RMSE={rmse:.2f}")

# Save the trained model
joblib.dump(model, "C:/Users/Owner/OneDrive/Documents/Utilization_and_Costs_of_Health_Services_for_Medicare_Fee-for-Service App/ml_model.pkl")
print("✅ Model and scaler saved successfully!")
