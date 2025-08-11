from django.shortcuts import render
import joblib
import numpy as np
import os

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'random_forest_optimized.pkl')
model = joblib.load(model_path)

# Feature order that the model expects
feature_order = [
    'Car Condition', 'Weather', 'Traffic Condition',
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'passenger_count', 'hour', 'day', 'month', 'weekday', 'year',
    'jfk_dist', 'ewr_dist', 'distance'
]

# Default values for unused features (example means)
default_values = {
    'pickup_longitude': -73.98,
    'pickup_latitude': 40.75,
    'dropoff_longitude': -73.98,
    'dropoff_latitude': 40.75,
    'passenger_count': 1,
    'hour': 12,
    'day': 15,
    'month': 6,
    'weekday': 3,
    'year': 2023,
    'jfk_dist': 15.0,
    'ewr_dist': 20.0,
    'distance': 1.0  # safe fallback if user doesn't input
}

def safe_float(value, default=0.0):
    """Convert value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def safe_int(value, default=0):
    """Convert value to int safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def home(request):
    if request.method == 'POST':
        # Get user inputs safely
        distance = safe_float(request.POST.get('distance'), default_values['distance'])
        car = safe_int(request.POST.get('car_condition'), 0)
        weather = safe_int(request.POST.get('weather'), 0)
        traffic = safe_int(request.POST.get('traffic'), 0)

        # Build features in correct order
        row = []
        for col in feature_order:
            if col == 'Car Condition':
                row.append(car)
            elif col == 'Weather':
                row.append(weather)
            elif col == 'Traffic Condition':
                row.append(traffic)
            elif col == 'distance':
                row.append(distance)
            else:
                row.append(default_values.get(col, 0))

        # Make prediction
        features = np.array(row).reshape(1, -1)
        prediction = model.predict(features)[0]

        return render(request, 'results.html', {'prediction': prediction})

    return render(request, 'home.html')
