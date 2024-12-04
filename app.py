from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Load models
short_form_model = load("optimized_logistic_regression_model.joblib")
long_form_model = load("logistic_regression_full_model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/short_form")
def short_form():
    return render_template("short_form.html")

@app.route("/long_form")
def long_form():
    return render_template("long_form.html")

@app.route("/predict_short", methods=["POST"])
def predict_short():
    try:
        features = np.array([
            request.form["age"],
            request.form["bmi"],
            request.form["physical_activity"],
            request.form["gender"],
            request.form["smoking_status"],
        ]).astype(float).reshape(1, -1)

        prediction = short_form_model.predict(features)[0]
        confidence = short_form_model.predict_proba(features).max()

        result = "Low Risk" if prediction == 0 else "High Risk"
        return render_template("result.html", prediction=result, confidence=f"{confidence:.2%}")
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict_long", methods=["POST"])
def predict_long():
    try:
        features = [
            float(request.form[key]) for key in request.form.keys()
        ]
        features = np.array(features).reshape(1, -1)

        prediction = long_form_model.predict(features)[0]
        confidence = long_form_model.predict_proba(features).max()

        result = "Low Risk" if prediction == 0 else "High Risk"
        return render_template("result.html", prediction=result, confidence=f"{confidence:.2%}")
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)


