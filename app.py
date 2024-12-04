from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load("optimized_logistic_regression_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)  # Ensure input is in the correct shape

    # Make predictions
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    # Return predictions as JSON
    return jsonify({
        "prediction": int(prediction[0]),
        "probability": probability.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))