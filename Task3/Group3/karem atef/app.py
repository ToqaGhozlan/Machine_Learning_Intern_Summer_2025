from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load('XGBClassifier.pkl')


required_columns = ['type of meal', 'repeated', 'lead time', 'market segment type', 'average price ', 'special requests']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'Missing required columns'})

        
        prediction = model.predict(df[required_columns])
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
