from django import forms


class FarePredictionForm(forms.Form):
    passenger_count = forms.IntegerField(min_value=1, max_value=10, required=True)
    distance = forms.FloatField(min_value=0.001, required=True)
    bearing = forms.FloatField(min_value=-3.14159, max_value=3.14159, required=True)

    Weather = forms.ChoiceField(choices=[
        ('sunny', 'Sunny'),
        ('cloudy', 'Cloudy'),
        ('rainy', 'Rainy'),
        ('stormy', 'Stormy'),
        ('windy', 'Windy')
    ], required=True)

    Traffic = forms.ChoiceField(choices=[
        ('Congested', 'Congested'),
        ('Dense', 'Dense'),
        ('Flow', 'Flow')
    ], required=True)

    car_condition = forms.ChoiceField(choices=[
        ('Excellent', 'Excellent'),
        ('Very Good', 'Very Good'),
        ('Good', 'Good'),
        ('Bad', 'Bad')
    ], required=True)

    # Optional fields
    user_name = forms.CharField(required=False)
    driver_name = forms.CharField(required=False)
    key = forms.CharField(required=False)
    pickup_datetime = forms.DateTimeField(required=False)
    pickup_longitude = forms.FloatField(required=False)
    pickup_latitude = forms.FloatField(required=False)
    dropoff_longitude = forms.FloatField(required=False)
    dropoff_latitude = forms.FloatField(required=False)

    hour = forms.IntegerField(min_value=0, max_value=23, required=True)
    weekday = forms.IntegerField(min_value=0, max_value=6, required=True)
    month = forms.IntegerField(min_value=1, max_value=12, required=False)
    year = forms.IntegerField(min_value=2000, max_value=2100, required=False)

    jfk_airport_distance = forms.FloatField(min_value=0, required=True)
    ewr_airport_distance = forms.FloatField(min_value=0, required=True)
    lga_airport_distance = forms.FloatField(min_value=0, required=True)
    statue_of_liberty_distance = forms.FloatField(min_value=0, required=True)
    nyc_center_distance = forms.FloatField(min_value=0, required=True)