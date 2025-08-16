from django.shortcuts import render
from .ml_pipeline import predict_fare
import pandas as pd
import numpy as np


def index(request):
    xgb_result = None
    linear_result = None
    knn_result = None
    ridge_result = None
    dtree_result = None
    error = None

    if request.method == "POST":
        try:
            # Collect exactly 9 numeric features
            input_features = [
                float(request.POST.get("pickup_longitude")),
                float(request.POST.get("pickup_latitude")),
                float(request.POST.get("dropoff_longitude")),
                float(request.POST.get("dropoff_latitude")),
                float(request.POST.get("passenger_count")),
                float(request.POST.get("hour")),
                float(request.POST.get("day")),
                float(request.POST.get("month")),
                float(request.POST.get("weekday")),
                float(request.POST.get("year")),
                float(request.POST.get("jfk_dist")),
                float(request.POST.get("ewr_dist")),
                float(request.POST.get("lga_dist")),
                float(request.POST.get("sol_dist")),
                float(request.POST.get("nyc_dist")),
                float(request.POST.get("distance")),
                float(request.POST.get("bearing ")),
            ]
            # marwan
            feature_names = [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "passenger_count",
                "hour",
                "day",
                "month",
                "weekday",
                "year",
                "jfk_dist",
                "ewr_dist",
                "lga_dist",
                "sol_dist",
                "nyc_dist",
                "distance",
                "bearing",
            ]
            features = np.array(input_features).reshape(1, -1)
            input_df = pd.DataFrame(features, columns=feature_names)
            # end for marwan editing
            # Pass directly
            xgb_pred, linear_pred = predict_fare(input_df)

            # Round results
            xgb_result = round(xgb_pred, 2)
            linear_result = round(linear_pred, 2)

        except Exception as e:
            error = f"Error11: {e}"

    return render(
        request,
        "myapp/index.html",
        {
            "xgb_result": xgb_result,
            "linear_result": linear_result,
            "error": error,
        },
    )
