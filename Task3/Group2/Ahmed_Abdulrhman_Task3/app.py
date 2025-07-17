import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('MLModel.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return render_template('index.html', prediction_text='Predicted Booking Status: {}'.format(int(prediction[0])))



if __name__ == "__main__":
    app.run(debug=True)