import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define dataset & model paths
model_path = r"C:\Cropify-Crop-Recommendation-System-main (1)\Cropify-Crop-Recommendation-System-main\Model\RDF_model.pkl"

# Load trained model
try:
    rdf_clf = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar UI
with st.sidebar:
    try:
        image = Image.open('logo_cropify.png')
        st.image(image, width=300)
    except Exception:
        st.warning("Logo not found!")

    st.markdown("## Crop Recommendation System")
    st.write("This model recommends the best crop based on soil and climate conditions.")

# Feature inputs
st.markdown("### Input Values for Prediction")

col1, col2 = st.columns(2)
with col1:
    n_input = st.number_input('Nitrogen (kg/ha):', min_value=0, max_value=140, step=1)
    p_input = st.number_input('Phosphorus (kg/ha):', min_value=5, max_value=145, step=1)
    k_input = st.number_input('Potassium (kg/ha):', min_value=5, max_value=205, step=1)
    temp_input = st.number_input('Temperature (°C):', min_value=9.0, max_value=43.0, step=0.1, format="%.2f")

with col2:
    hum_input = st.number_input('Humidity (%):', min_value=15.0, max_value=99.0, step=0.1, format="%.2f")
    ph_input = st.number_input('pH:', min_value=3.6, max_value=9.9, step=0.1, format="%.2f")
    rain_input = st.number_input('Rainfall (mm):', min_value=21.0, max_value=298.0, step=0.1, format="%.2f")

# Ensure only 7 features are passed to the model
predict_inputs = pd.DataFrame([[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input]],
                              columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

# Prediction button
if st.button('Recommend Crop'):
    try:
        prediction = rdf_clf.predict(predict_inputs)[0]
        st.success(f"🌱 Recommended Crop: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
