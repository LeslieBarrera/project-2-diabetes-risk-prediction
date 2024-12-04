from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the optimized logistic regression model
short_form_model = load("optimized_logistic_regression_model.joblib")

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        data = request.form

        # Map the user inputs to the required features
        features = [
            int(data.get("age", 0)),
            float(data.get("bmi", 0)),
            1 if data.get("physical_activity") == "Active" else 0,
            1 if data.get("gender") == "Male" else 0,
            1 if data.get("smoking_status") == "Smoker" else 0,
        ]

        # Reshape for model prediction
        features_array = np.array(features).reshape(1, -1)

        # Make predictions using the logistic regression model
        prediction = short_form_model.predict(features_array)
        probability = short_form_model.predict_proba(features_array)

        # Prepare the results
        result = {
            "risk": "Low Risk" if prediction[0] == 0 else "High Risk",
            "confidence": f"{max(probability[0]) * 100:.2f}%"
        }

        return render_template("result.html", result=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

