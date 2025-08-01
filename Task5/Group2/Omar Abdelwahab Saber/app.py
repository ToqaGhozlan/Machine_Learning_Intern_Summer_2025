import streamlit as st
import pandas as pd
import pickle

st.title("Taxi Fare Prediction")
st.write("This app predicts the taxi fare amount based on various trip and environment details.")
st.image("https://oceansidetaxi.ca/wp-content/uploads/uber-estimate.png", width=700)

# --- User Inputs ---
car_condition = st.selectbox("Car Condition", ["Bad", "Good", "Very Good", "Excellent"])
car_map = {"Bad": 0, "Good": 1, "Very Good": 2, "Excellent": 3}
car_condition_val = car_map[car_condition]

passenger_count = st.number_input("Number of Passengers", min_value=0, max_value=6, value=1)

hour = st.slider("Hour of the Day", 0, 23, 12)
day = st.slider("Day of Month", 1, 31, 15)
month = st.selectbox("Month", list(range(1, 13)))
weekday_name = st.selectbox("Day of the Week", [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
])
weekday_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
weekday = weekday_map[weekday_name]
year = st.selectbox("Year", [2009,2010,2011,2012,2013,2014,2015])

sol_dist = st.number_input("Distance from statue of liberty (km)", min_value=0.0, max_value=50.0, step=0.1)
distance = st.number_input("Trip Distance (km)", min_value=0.0, max_value=50.0, step=0.1)

# Weather (cloudy is dropped in one-hot, so not shown)
weather = st.selectbox("Weather", ["Cloudy", "Rainy", "Stormy", "Sunny", "Windy"])
weather_features = ['weather_rainy', 'weather_stormy', 'weather_sunny', 'weather_windy']
weather_encoding = [1 if weather.lower() in w else 0 for w in weather_features]

# Traffic (Congested Traffic is dropped)
traffic = st.selectbox("Traffic Condition", ["Congested Traffic", "Dense Traffic", "Flow Traffic"])
traffic_features = ['traffic_condition_Dense Traffic', 'traffic_condition_Flow Traffic']
traffic_encoding = [1 if traffic in t else 0 for t in traffic_features]

# --- Prepare Input ---
input_data = [[
    car_condition_val, passenger_count, hour, day, month, weekday,
    year, sol_dist, distance
] + weather_encoding + traffic_encoding]

columns = [
    'car_condition', 'passenger_count', 'hour', 'day', 'month', 'weekday',
    'year', 'sol_dist', 'distance',
    'weather_rainy', 'weather_stormy', 'weather_sunny', 'weather_windy',
    'traffic_condition_Dense Traffic', 'traffic_condition_Flow Traffic'
]

input_df = pd.DataFrame(input_data, columns=columns)

# --- Load Scaler and Model ---
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# --- Scale sol_dist and distance only ---
input_df[["sol_dist", "distance"]] = scaler.transform(input_df[["sol_dist", "distance"]])

# --- Predict ---
if st.button("Predict Fare"):
    predicted_fare = model.predict(input_df)[0]
    st.write("**Predicted Fare Amount:**")
    st.success(f"${predicted_fare:.2f}")
    st.write("---")
    st.write("**Input Data Used for Prediction:**")
    st.dataframe(input_df)
