from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    
    return render_template('index.html', prediction_text="📢 Prediction done (placeholder)")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
