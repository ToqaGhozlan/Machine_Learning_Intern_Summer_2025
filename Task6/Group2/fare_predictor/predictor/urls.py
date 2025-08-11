from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.fare_predict_view, name='fare_predict'),
]
