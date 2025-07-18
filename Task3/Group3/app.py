import streamlit as st
import joblib
import numpy as np

# Load model and features
with open('booking_model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = joblib.load(f)

st.title("Booking Status Prediction")
st.write("Enter the features below:")

# Create input fields dynamically
user_input = []
for col in feature_columns:
    val = st.number_input(f"{col}", step=0.01)
    user_input.append(val)

if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    result = "Cancelled" if prediction == 1 else "Not Cancelled"
    st.success(f"Booking Status: {result}")
