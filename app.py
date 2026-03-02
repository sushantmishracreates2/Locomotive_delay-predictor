import os
import joblib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🚆 Locomotive Delay Prediction System")

st.subheader("Enter Train Details")

# Input fields
speed_kmph = st.number_input("Speed (km/h)", 40, 150, 80)
congestion_level = st.slider("Congestion Level", 0.0, 1.0, 0.5)
route_deviation_meters = st.number_input("Route Deviation (meters)", 0, 1000, 100)
distance_to_next_station_km = st.number_input("Distance to Next Station (km)", 1, 100, 20)
fuel_consumption_lph = st.number_input("Fuel Consumption (L/h)", 100, 500, 250)

track_condition = st.selectbox("Track Condition", ["Good", "Moderate", "Poor"])
weather = st.selectbox("Weather", ["Clear", "Rain", "Fog", "Storm"])
signal_status = st.selectbox("Signal Status", ["Green", "Yellow", "Red"])

scheduled_arrival_hour = st.slider("Scheduled Arrival Hour", 0, 23, 12)

# Manual encoding (must match training encoding)
track_map = {"Good": 0, "Moderate": 1, "Poor": 2}
weather_map = {"Clear": 0, "Fog": 1, "Rain": 2, "Storm": 3}
signal_map = {"Green": 0, "Red": 1, "Yellow": 2}

input_data = pd.DataFrame([{
    "train_id": 1001,
    "current_lat": 20.0,
    "current_lon": 78.0,
    "destination_lat": 25.0,
    "destination_lon": 80.0,
    "speed_kmph": speed_kmph,
    "track_condition": track_map[track_condition],
    "weather": weather_map[weather],
    "congestion_level": congestion_level,
    "signal_status": signal_map[signal_status],
    "distance_to_next_station_km": distance_to_next_station_km,
    "route_deviation_meters": route_deviation_meters,
    "fuel_consumption_lph": fuel_consumption_lph,
    "scheduled_arrival_hour": scheduled_arrival_hour
}])

if st.button("Predict Delay"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Train is likely to be DELAYED")
        st.write(f"Confidence: {round(probability * 100, 2)}%")
    else:
        st.success("✅ Train is likely to be ON TIME")
        st.write(f"Confidence: {round((1-probability) * 100, 2)}%")