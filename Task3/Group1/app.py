import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


def month_arr(date):
    parts = date.split("/")
    if len(parts) == 1:
        parts = date.split("-")
        mon = parts[1]
        day = parts[2]
    else:
        mon = parts[0]
        day = parts[1]
    return int(mon), int(day)


def more_than_year(data):
    year = 0
    while data > 12:
        data -= 12
        year += 1
    return data


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Get raw inputs
        date_of_res = request.form["date of reservation"]
        lead_time = int(request.form["lead time"])
        avg_price = float(request.form["average price"])
        car_parking_space = int(request.form["car parking space"])
        number_of_adults = int(request.form["number of adults"])
        number_of_children = int(request.form["number of children"])
        special_requests = int(request.form["special requests"])
        repeated = int(request.form["repeated"])
        number_of_weekend_nights = int(request.form["number of weekend nights"])
        number_of_week_nights = int(request.form["number of week nights"])

        # Categorical inputs
        meal = request.form["type of meal"]
        room = request.form["room type"]
        market = request.form["market segment type"]

        # 2. Extract date parts
        month, day = month_arr(date_of_res)
        month_of_arrive = more_than_year(month + (day + lead_time) // 30)

        # 3. One-hot encode categorical vars
        # All possible encoded dummy columns:
        all_meals = ["Meal Plan 1", "Meal Plan 2"]
        all_rooms = ["Room_Type 1", "Room_Type 6", "Room_Type 7"]
        all_markets = ["Complementary", "Corporate", "Online"]

        meal_dict = {f"type of meal_{m}": 0 for m in all_meals}
        room_dict = {f"room type_{r}": 0 for r in all_rooms}
        market_dict = {f"market segment type_{m}": 0 for m in all_markets}

        if f"type of meal_{meal}" in meal_dict:
            meal_dict[f"type of meal_{meal}"] = 1
        if f"room type_{room}" in room_dict:
            room_dict[f"room type_{room}"] = 1
        if f"market segment type_{market}" in market_dict:
            market_dict[f"market segment type_{market}"] = 1

        # 4. Build input DataFrame
        input_dict = {
            "number of adults": number_of_adults,
            "number of children": number_of_children,
            "number of weekend nights": number_of_weekend_nights,
            "number of week nights": number_of_week_nights,
            "car parking space": car_parking_space,
            "lead time": lead_time,
            "repeated": repeated,
            "average price ": avg_price,
            "special requests": special_requests,
            "month of arrive": month_of_arrive,
        }
        input_dict.update(meal_dict)
        input_dict.update(room_dict)
        input_dict.update(market_dict)

        df_input = pd.DataFrame([input_dict])

        # 5. Scale input (IMPORTANT: use same scaler fitted on training data)
        # Ideally, load the fitted scaler from pickle. Here we simulate using training features:
        # Replace this with loading your saved scaler if you have one.
        scaler = pickle.load(open("scaler.pkl", "rb"))
        df_scaled = scaler.transform(df_input)

        # 6. Predict
        prediction = model.predict(df_scaled)[0]

        result = "Booking Not Canceled" if prediction == 1 else "Booking Canceled"
        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    flask_app.run(debug=True)
