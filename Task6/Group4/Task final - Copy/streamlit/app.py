import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler


model = joblib.load("mode2.pkl")
scaler = joblib.load("scaler.pkl")

weather_map = {"sunny": 1, "cloudy": 2, "stormy": 3, "rainy": 4, "windy": 5}
car_condition_map = {"Very Good": 1, "Bad": 2, "Excellent": 3, "Good": 4}
st.title("Expect the price of the trip")

Car_Condition_str = st.selectbox("Car Condition", list(car_condition_map.keys()))
Weather_str = st.selectbox("Weather", list(weather_map.keys()))

Car_Condition = car_condition_map[Car_Condition_str]
Weather = weather_map[Weather_str]

passenger_count = st.number_input("Passenger Count", min_value=1)
hour = st.number_input("Hour", min_value=0, max_value=23)
day = st.number_input("Day", min_value=1, max_value=31)
month = st.number_input("Month", min_value=1, max_value=12)
weekday = st.number_input("Weekday", min_value=0, max_value=6)
year = st.number_input("Year", min_value=2000)

distance = st.number_input("Distance")

distance =  scaler.transform([[distance]])[0][0]

traffic_options = ["Congested", "Flow", "Dense"]
traffic_choice = st.selectbox("Traffic Condition", traffic_options)


if traffic_choice == "Congested":
    Congested_Traffic, Flow_Traffic, Dense_Traffic = 1, 0, 0
elif traffic_choice == "Flow":
    Congested_Traffic, Flow_Traffic, Dense_Traffic = 0, 1, 0
else: 
    Congested_Traffic, Flow_Traffic, Dense_Traffic = 0, 0, 1

if st.button("pridect"):
    input_data = np.array([[Car_Condition, Weather, passenger_count, hour, day, month, weekday, year, distance, Congested_Traffic, Flow_Traffic, Dense_Traffic]])
    prediction = model.predict(input_data)
    st.success(f"fare amount : {prediction[0]:.2f}")
