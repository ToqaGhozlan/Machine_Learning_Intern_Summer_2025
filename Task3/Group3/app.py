from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)


model = joblib.load('rf_model.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        reservation_date = datetime.strptime(data.get("reservation_date"), "%Y-%m-%d")
        reservation_day = reservation_date.day
        reservation_month = reservation_date.month
        reservation_weekday = reservation_date.weekday()

        weekend_nights = int(data.get("weekend_nights", 0))
        week_nights = int(data.get("week_nights", 0))
        total_nights = weekend_nights + week_nights
        weekends_to_total = weekend_nights / total_nights if total_nights > 0 else 0

        input_dict = {
            'car parking space': int(data.get("car_parking", 0)),
            'days from booking to arrival': int(data.get("lead_time", 0)),
            'visited before': int(data.get("visited_before", 0)),
            'average price': float(data.get("average_price", 0.0)),
            'special requests': int(data.get("special_requests", 0)),
            'reservation_day': reservation_day,
            'reservation_month': reservation_month,
            'reservation_weekday': reservation_weekday,
            'weekends to total': weekends_to_total
        }

        categorical = [
            data.get("type_of_meal", "Not Selected"),
            data.get("room_type", "Room_Type 1"),
            data.get("market_type", "Online")
        ]

        cat_encoded = encoder.transform([categorical])
        if hasattr(cat_encoded, "toarray"):
            cat_encoded = cat_encoded.toarray()[0]  

        final_input = list(input_dict.values()) + list(cat_encoded)
        final_input = np.array(final_input).reshape(1, -1)

        print("Final input shape:", final_input.shape)

        prediction = model.predict(final_input)[0]
        proba = model.predict_proba(final_input)[0]

        confidence = proba[prediction]

        # Override logic: if confidence < 0.7, predict "Canceled"
        if confidence < 0.7:
            label = "Canceled"
        else:
            label = "Canceled" if prediction == 1 else "Not Canceled"

        return jsonify({
            "prediction": label,
            "confidence": float(confidence)
        })


    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
