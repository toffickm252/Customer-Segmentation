import streamlit as st
import joblib
import pandas as pd
import os

# Function to load model and scaler with improved error handling
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_file = os.path.join(base_dir, "..", "models", "kmeans_model.joblib")
    scaler_file = os.path.join(base_dir, "..", "models", "scaler.joblib")

    # Resolve absolute paths
    model_file = os.path.abspath(model_file)
    scaler_file = os.path.abspath(scaler_file)

    # Debug print for Streamlit Cloud
    st.write("Model path:", model_file)
    st.write("Scaler path:", scaler_file)

    # Check existence BEFORE loading
    if not os.path.exists(model_file):
        st.error("Model file missing.")
        st.stop()

    if not os.path.exists(scaler_file):
        st.error("Scaler file missing.")
        st.stop()

    # Load both
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error(f"Loading error: {e}")
        st.stop()


model ,scaler= load_model()
# Title
st.title("Customer Segmentation Predictor")

# Input Forms
recency = st.number_input("Days Since Last Purchase (Recency)", min_value=0, value=10)
frequency = st.number_input("Total Number of Orders (Frequency)", min_value=0, value=5)
monetary = st.number_input("Total Amount Spent (Monetary)", min_value=0.0, value=100.0)