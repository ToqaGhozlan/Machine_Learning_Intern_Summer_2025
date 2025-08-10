import joblib
import numpy as np
from django.shortcuts import render

model = joblib.load('predictor/fare_model.pkl')
scaler = joblib.load('predictor/scaler.pkl')

def home(request):
    return render(request, 'predictor/home.html')

def predict(request):
    if request.method == 'POST':
        try:
            distance = float(request.POST['distance'])
            hour = int(request.POST['hour'])
            weekday = int(request.POST['weekday'])
            passenger_count = int(request.POST['passenger'])

            data = np.array([[distance, hour, weekday, passenger_count]])
            data_scaled = scaler.transform(data)
            fare = model.predict(data_scaled)[0]
            fare = round(fare, 2)

            return render(request, 'predictor/home.html', {'result': fare})
        except Exception as e:
            return render(request, 'predictor/home.html', {'error': str(e)})
    return render(request, 'predictor/home.html')
