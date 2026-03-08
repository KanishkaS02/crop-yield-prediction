import streamlit as st
import pickle
import numpy as np

# load model
model = pickle.load(open("model/yield_model.pkl","rb"))

st.title("AI Crop Yield Prediction 🌾")

st.write("Enter the farm details below")

rainfall = st.number_input("Rainfall")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
fertilizer = st.number_input("Fertilizer")

if st.button("Predict Yield"):

    input_data = np.array([[rainfall,temperature,humidity,fertilizer]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Crop Yield: {prediction[0]:.2f} tons/hectare")