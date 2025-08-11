from django.shortcuts import render , redirect
from .forms import FarePredictionForm
from . import pipeline


def home(request):
    return render(request, 'predictor/index.html')

def fare_predict_view(request):
    
    if request.method == "POST":
        form = FarePredictionForm(request.POST)
        if form.is_valid():
            cleaned_data = form.cleaned_data
            print(cleaned_data.keys())
            X = pipeline.preprocess_input(cleaned_data)
            print(X.columns)
            X = pipeline.feature_engineering(X)
            X = pipeline.feature_encoding(X)
            X = pipeline.prepare_features(X)

            gb_prediction = pipeline.predict_fare_gb(X)
            gxb_prediction = pipeline.predict_fare_gxb(X)   
            rf_prediction = pipeline.predict_fare_rf(X)

            avg_prediction = round((pipeline.predict_fare_gb(X) + pipeline.predict_fare_gxb(X) + pipeline.predict_fare_rf(X)) / 3,2)
    
            context = {
                'predictions': {
                    'Gradient Boosting': gb_prediction,
                    'XGBoost': gxb_prediction,
                    'Random Forest': rf_prediction,
                },
                'avg_prediction': avg_prediction
            }
            return render(request, 'predictor/pred_results.html', context)
    return redirect('home')
