from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load models
short_form_model = load("optimized_logistic_regression_model.joblib")
long_form_model = load("long_form_model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/short_form", methods=["GET", "POST"])
def short_form():
    if request.method == "POST":
        # Get form data for short form
        age = int(request.form["age"])
        bmi = float(request.form["bmi"])
        physical_activity = 1 if request.form["physical_activity"] == "Active" else 0
        gender = 1 if request.form["gender"] == "Male" else 0
        smoking_status = 1 if request.form["smoking_status"] == "Smoker" else 0

        # Predict using short form model
        features = np.array([age, bmi, physical_activity, gender, smoking_status]).reshape(1, -1)
        prediction = short_form_model.predict(features)
        probability = short_form_model.predict_proba(features)[0]

        risk = "High Risk" if prediction[0] == 1 else "Low Risk"

        return render_template("result.html", risk=risk, probability=probability, form="Short Form")
    return render_template("short_form.html")

@app.route("/long_form", methods=["GET", "POST"])
def long_form():
    if request.method == "POST":
        # Gather all inputs for long form
        features = [float(request.form[key]) for key in request.form.keys()]
        features = np.array(features).reshape(1, -1)

        # Predict using long form model
        prediction = long_form_model.predict(features)
        probability = long_form_model.predict_proba(features)[0]

        risk = "High Risk" if prediction[0] == 1 else "Low Risk"

        return render_template("result.html", risk=risk, probability=probability, form="Long Form")
    return render_template("long_form.html")

if __name__ == "__main__":
    app.run(debug=True)

