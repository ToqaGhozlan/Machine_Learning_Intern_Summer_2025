from django.shortcuts import render
import joblib
import numpy as np

# تحميل الموديل والـ scaler
model = joblib.load("predictor/mode2.pkl")
scaler = joblib.load("predictor/scaler.pkl")

weather_map = {"sunny": 1, "cloudy": 2, "stormy": 3, "rainy": 4, "windy": 5}
car_condition_map = {"Very Good": 1, "Bad": 2, "Excellent": 3, "Good": 4}
traffic_options = ["Congested", "Flow", "Dense"]

def predict_price(request):
    result = None
    if request.method == "POST":
        car_condition_str = request.POST.get("car_condition")
        weather_str = request.POST.get("weather")
        passenger_count = int(request.POST.get("passenger_count"))
        hour = int(request.POST.get("hour"))
        day = int(request.POST.get("day"))
        month = int(request.POST.get("month"))
        weekday = int(request.POST.get("weekday"))
        year = int(request.POST.get("year"))
        distance = float(request.POST.get("distance"))
        traffic_choice = request.POST.get("traffic")

        # تحويل النصوص للأرقام
        Car_Condition = car_condition_map[car_condition_str]
        Weather = weather_map[weather_str]
        distance_scaled = scaler.transform([[distance]])[0][0]

        # One-hot encoding لحالة المرور
        if traffic_choice == "Congested":
            Congested_Traffic, Flow_Traffic, Dense_Traffic = 1, 0, 0
        elif traffic_choice == "Flow":
            Congested_Traffic, Flow_Traffic, Dense_Traffic = 0, 1, 0
        else:
            Congested_Traffic, Flow_Traffic, Dense_Traffic = 0, 0, 1

        # تجهيز البيانات للتنبؤ
        input_data = np.array([[Car_Condition, Weather, passenger_count, hour, day, month,
                                 weekday, year, distance_scaled,
                                 Congested_Traffic, Flow_Traffic, Dense_Traffic]])
        prediction = model.predict(input_data)
        result = round(prediction[0], 2)

    return render(request, "predictor/form.html", {
        "result": result,
        "car_condition_choices": car_condition_map.keys(),
        "weather_choices": weather_map.keys(),
        "traffic_choices": traffic_options
    })
