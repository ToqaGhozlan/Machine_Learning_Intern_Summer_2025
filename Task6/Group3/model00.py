from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Verify model files exist
if not os.path.exists('random_forest_model.pkl'):
    raise FileNotFoundError("Model file not found")
if not os.path.exists('standard_scaler.pkl'):
    raise FileNotFoundError("Scaler file not found")

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('standard_scaler.pkl')

# Define the exact feature order expected by the model
FEATURE_ORDER = [
    'distance',
    'year',
    'month',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate input data
        data = request.get_json()
        
        # Create input array in EXACT order expected by model
        input_data = np.array([[
            float(data['distance']),
            int(data['year']),
            int(data['month']),
            float(data['pickup_longitude']),
            float(data['pickup_latitude']),
            float(data['dropoff_longitude']),
            float(data['dropoff_latitude'])
        ]])
        
        # Scale features
        scaled_data = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(scaled_data)
        
        return jsonify({
            'fare_amount': float(prediction[0])
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)