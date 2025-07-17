from flask import Flask, render_template, request, redirect, url_for
import traceback
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('hotel_booking_polynomial_logistic_model.joblib')

# Replace this list with the exact features used during model training
expected_features = [
    'adults', 'children', 'weekend_nights', 'total_nights', 'lead_time_log',
    'cancelled_times', 'not_cancelled_times', 'avg_price', 'car_parking',
    'is_repeated', 'special_requests_log',
    'meal_type_Buffet', 'meal_type_Meal Plan 1', 'meal_type_Meal Plan 2', 'meal_type_Meal Plan 3',
    'room_type_Room_Type 1', 'room_type_Room_Type 2', 'room_type_Room_Type 3',
    'market_type_Online', 'market_type_Offline', 'market_type_Corporate',
    'market_type_Complementary', 'market_type_Other'
]

@app.route('/')
def index():
    prediction = request.args.get("prediction")
    probability = request.args.get("probability")
    return render_template("index.html", prediction=prediction, probability=probability)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_fields = [
            'adults', 'children', 'weekend_nights', 'lead_time',
            'cancelled_times', 'not_cancelled_times', 'avg_price',
            'car_parking', 'is_repeated', 'special_requests',
            'meal_type', 'room_type', 'market_type'
        ]

        data = {}
        errors = {}

        for field in input_fields:
            value = request.form.get(field)
            if not value:
                errors[field] = "This field is required."
                continue
            try:
                if field in ['meal_type', 'room_type', 'market_type']:
                    data[field] = value
                elif field in ['avg_price']:
                    data[field] = float(value)
                else:
                    data[field] = int(value)
            except ValueError:
                errors[field] = "Invalid number."

        if data.get("lead_time", 0) < 0:
            errors["lead_time"] = "Lead time must be >= 0"
        if data.get("avg_price", 0) < 0:
            errors["avg_price"] = "Price must be >= 0"

        if errors:
            return render_template("index.html", error="Some fields are invalid.", errors=errors, request=request)

        # 2. Create raw DataFrame
        df = pd.DataFrame([data])

        # 3. Feature Engineering
        df["total_nights"] = df["weekend_nights"]  # 'week_nights' was removed earlier
        df["lead_time_log"] = np.log1p(df["lead_time"])
        df["special_requests_log"] = np.log1p(df["special_requests"])
        df.drop(columns=["lead_time", "special_requests"], inplace=True)

        # 4. One-hot encoding
        df = pd.get_dummies(df, columns=["meal_type", "room_type", "market_type"])

        # 5. Add any missing columns (from training)
        expected_features = [
            'adults', 'children', 'weekend_nights', 'car_parking', 'is_repeated',
            'cancelled_times', 'not_cancelled_times', 'avg_price',
            'total_nights',
            'meal_type_Meal Plan 1', 'meal_type_Meal Plan 2', 'meal_type_Meal Plan 3', 'meal_type_Not Selected',
            'room_type_Room_Type 1', 'room_type_Room_Type 2', 'room_type_Room_Type 3',
            'room_type_Room_Type 4', 'room_type_Room_Type 5', 'room_type_Room_Type 6', 'room_type_Room_Type 7',
            'market_type_Aviation', 'market_type_Complementary', 'market_type_Corporate',
            'market_type_Offline', 'market_type_Online',
            'lead_time_log', 'special_requests_log'
        ]

        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # 6. Reorder to match training
        df = df[expected_features]

        # 7. Predict
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][prediction]
        result = "Confirmed ✅" if prediction == 0 else "Canceled ❌"

        return redirect(url_for("index", prediction=result, probability=f"{proba:.2f}"))

    except Exception as e:
        traceback.print_exc()
        return render_template("index.html", error="Something went wrong. Please try again.", errors={})

if __name__ == '__main__':
    app.run(debug=True)
