import streamlit as st
import pandas as pd
import pickle

# Load trained model and label encoder
model = pickle.load(open("crop_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("🌱 Smart Crop Recommendation System")
st.write("Enter soil and weather details to get the best crop recommendation")

# Input fields
nitrogen = st.number_input("Nitrogen", min_value=0.0)
phosphorus = st.number_input("Phosphorus", min_value=0.0)
potassium = st.number_input("Potassium", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph_value = st.number_input("pH Value", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Predict button
if st.button("Predict Crop"):
    total_nutrients = nitrogen + phosphorus + potassium
    temp_humidity = temperature * humidity

    input_data = pd.DataFrame([{
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "Temperature": temperature,
        "Humidity": humidity,
        "pH_Value": ph_value,
        "Rainfall": rainfall,
        "Total_Nutrients": total_nutrients,
        "Temp_Humidity": temp_humidity
    }])

    prediction = model.predict(input_data)
    crop_name = le.inverse_transform(prediction)[0]

    st.success(f"Recommended Crop: {crop_name}")

    st.info(f"""
Input Summary:
Nitrogen: {nitrogen}
Phosphorus: {phosphorus}
Potassium: {potassium}
Temperature: {temperature} °C
Humidity: {humidity} %
pH Value: {ph_value}
Rainfall: {rainfall} mm
""")
