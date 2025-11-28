import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st


# -------------------- LOAD MODEL --------------------
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "..", "models")

    model_path = os.path.join(model_dir, "kmeans_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")

    if not os.path.exists(model_path):
        st.error(f"Missing model file: {model_path}")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error(f"Missing scaler file: {scaler_path}")
        st.stop()

    return joblib.load(model_path), joblib.load(scaler_path)


model, scaler = load_model()


# -------------------- UI --------------------
st.title("🛍️ Customer Segmentation Predictor")
st.markdown("Enter customer details below:")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input("Days Since Last Purchase", min_value=0, value=10)

with col2:
    frequency = st.number_input("Total Orders", min_value=1, value=5)

with col3:
    monetary = st.number_input("Total Spent ($)", min_value=0.01, value=100.0)


# -------------------- PREDICTION --------------------
if st.button("Predict Segment"):

    # Log transforms
    data = pd.DataFrame({
        "Recency": [np.log(recency + 1)],
        "Frequency": [np.log(frequency)],
        "Monetary": [np.log(monetary)]
    })

    scaled = scaler.transform(data)
    cluster = model.predict(scaled)[0]

    st.markdown("---")
    st.subheader("Prediction Result:")

    if cluster == 1:
        st.success("🏆 Cluster 1: VIP Customer")
        st.write("Action: Send exclusive offers.")
    else:
        st.warning("💤 Cluster 0: Hibernating Customer")
        st.write("Action: Send a re-engagement discount.")
