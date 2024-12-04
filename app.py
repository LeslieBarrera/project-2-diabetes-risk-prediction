from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load("optimized_logistic_regression_model.joblib")

@app.route('/')
def home():
    return "Welcome to the Diabetes Risk Prediction API. Use the /predict endpoint for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return jsonify({
        "prediction": int(prediction[0]),
        "probability": probability.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)