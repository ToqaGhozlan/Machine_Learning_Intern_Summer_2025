from django.shortcuts import render
import joblib
import numpy as np

model = joblib.load("random_forest_model.pkl")

def home(request):
    if request.method == "POST":
        try:
            pickup_longitude = float(request.POST['feature2'])
            pickup_latitude = float(request.POST['feature3'])
            dropoff_longitude = float(request.POST['feature4'])
            dropoff_latitude = float(request.POST['feature5'])
            passenger_count = float(request.POST['feature6'])
            hour = float(request.POST['feature7'])
            day = float(request.POST['feature8'])
            weekday = float(request.POST['feature9'])
            year = float(request.POST['feature10'])
            jfk_dist = float(request.POST['feature11'])
            ewr_dist = float(request.POST['feature12'])
            lga_dist = float(request.POST['feature13'])
            sol_dist = float(request.POST['feature14'])
            nyc_dist = float(request.POST['feature15'])
            distance = float(request.POST['feature16'])
            bearing = float(request.POST['feature17'])
            weekday_name = float(request.POST['feature18'])  
            car_condition_type = float(request.POST['feature19'])  
            traffic_condition_type = float(request.POST['feature20'])  
            weather_type = float(request.POST['feature21'])  

            input_data = np.array([[
                pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
                passenger_count, hour, day, weekday, year,
                jfk_dist, ewr_dist, lga_dist, sol_dist, nyc_dist,
                distance, bearing, weekday_name,
                car_condition_type, traffic_condition_type, weather_type
            ]])

            prediction = model.predict(input_data)[0]

            return render(request, "predictor/home.html", {
                "prediction": round(prediction, 2)
            })

        except Exception as e:
            return render(request, "predictor/home.html", {
                "error": f"Invalid input. Please enter numeric values. ({str(e)})"
            })

    return render(request, "predictor/home.html")
