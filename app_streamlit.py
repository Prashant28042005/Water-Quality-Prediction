import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("water_quality_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💧 Water Quality Prediction App")

# Input fields
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
solids = st.number_input("Solids", min_value=0.0, value=20000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=330.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=420.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=11.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=65.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=3.0)

# Collect features
features = np.array([[ph, hardness, solids, chloramines, sulfate,
                      conductivity, organic_carbon, trihalomethanes, turbidity]])

# Apply scaling
features_scaled = scaler.transform(features)

# Prediction
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    if prediction == 1:
        st.success("✅ Water is Drinkable")
    else:
        st.error("❌ Water is Not Drinkable")
