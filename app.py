from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load("optimized_logistic_regression_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [
            float(request.form["feature1"]),
            float(request.form["feature2"]),
            float(request.form["feature3"]),
            float(request.form["feature4"]),
            float(request.form["feature5"]),
        ]

        # Convert features to numpy array for prediction
        features_array = np.array(features).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)

        # Return results to be displayed
        return render_template(
            'result.html',
            prediction=int(prediction[0]),
            probability=probability[0]
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
