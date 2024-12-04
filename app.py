from flask import Flask, request, render_template, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load models
short_model = load("optimized_logistic_regression_model.joblib")
long_model = load("long_form_model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form", methods=["GET"])
def form():
    form_type = request.args.get("type")  # Get form type from query parameters
    if form_type == "short":
        return render_template("short_form.html")
    elif form_type == "long":
        return render_template("long_form.html")
    else:
        return "Invalid form type", 400

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    form_type = data.get("form_type")  # short or long
    features = np.array(data["features"]).reshape(1, -1)

    if form_type == "short":
        model = short_model
        confidence = "70%"
    elif form_type == "long":
        model = long_model
        confidence = "85%"
    else:
        return jsonify({"error": "Invalid form type"}), 400

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features).tolist()

    # Translate prediction to user-friendly labels
    risk_level = "Low Risk" if prediction == 0 else "High Risk"

    return jsonify({
        "form_type": form_type,
        "risk_level": risk_level,
        "confidence": confidence,
        "probabilities": probabilities
    })

if __name__ == "__main__":
    app.run(debug=True)

