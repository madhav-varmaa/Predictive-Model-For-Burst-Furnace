import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load your dataset (assuming 'bf3_co_co2_input.csv' contains your data)
df = pd.read_csv('bf3_co_co2_input.csv')
df = df.dropna()

# Separate features (x) and target (y)
x = df.iloc[:, :13]
y = df.iloc[:, 15]

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)

# User input function
def user_input():
    cb_flow = st.number_input("Enter CB_FLOW")
    cb_press = st.number_input("Enter CB_PRESS")
    steam_flow = st.number_input("Enter STEAM_FLOW")
    steam_temp = st.number_input("Enter STEAM_TEMP")
    steam_press = st.number_input("Enter STEAM_PRESS")
    O2_press = st.number_input("Enter O2_PRESS")
    O2_flow = st.number_input("Enter O2_FLOW")
    O2_per = st.number_input("Enter O2_PER")
    pci = st.number_input("Enter PCI")
    atm = st.number_input("Enter ATM_HUMID")
    hb_press = st.number_input("Enter HB_PRESS")
    spray = st.number_input("Enter TOP_SPRAY")
    h2 = st.number_input("Enter H2")
    # ... (add other input features here)

    data = {
        'CB_FLOW': cb_flow,
        'CB_PRESS': cb_press,
        'STEAM_FLOW': steam_flow,
        'STEAM_TEMP': steam_temp,
            'STEAM_PRESS': steam_press,
            'O2_PRESS': O2_press,
            'O2_FLOW': O2_flow,
            'O2_PER': O2_per,
            'PCI': pci,
            'ATM_HUMID': atm,
            'HB_PRESS': hb_press,
            'TOP_SPRAY' : spray,
            'H2' : h2
        # ... (add other features)
    }
    inputs = pd.DataFrame(data, index=[0])
    return inputs

    
# Get user input
input_df = user_input()

# Display prediction
if st.button("PREDICT"):

    # Make predictions
    prediction = model.predict(scaler.transform(input_df))
    st.success(f"THE PREDICTED CO/CO2 RATIO is: {prediction[0]:.6f}")

    # Optional: Display feature importances
    feature_importances = model.feature_importances_
    st.bar_chart(pd.Series(feature_importances, index=x.columns))

    # Optional: Display evaluation metrics (R2, MSE, MAE)
    y_pred = model.predict(x_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"R2 score: {r2score:.6f}")
    st.write(f"Mean squared error: {mse:.6f}")
    st.write(f"Mean absolute error: {mae:.6f}")
