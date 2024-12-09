from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
short_form_model = load("diabetes_short_form_model.pkl")
long_form_model = load("diabetes_long_form_model.pkl")  # Updated to match your saved model file

# Load short form features
short_form_features = pd.read_csv("diabetes_short_form_features.csv")["Feature"].tolist()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/short_form", methods=["GET", "POST"])
def short_form():
    if request.method == "POST":
        # Collect input for short form
        inputs = {}
        for feature in short_form_features:
            inputs[feature] = float(request.form.get(feature, 0))  # Default to 0 if missing
        
        # Convert inputs to a DataFrame with feature names
        features_df = pd.DataFrame([inputs])

        # Predict using the short form model
        prediction = short_form_model.predict(features_df)
        probability = short_form_model.predict_proba(features_df)[0]

        # Map prediction to risk
        classes = {0: "Non-Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}
        risk = classes[prediction[0]]

        return render_template(
            "result.html", 
            risk=risk, 
            probability={
                "Non-Diabetes": round(probability[0], 4),
                "Pre-Diabetes": round(probability[1], 4),
                "Diabetes": round(probability[2], 4),
            }, 
            form="Short Form"
        )
    return render_template("short_form.html", features=short_form_features)

@app.route("/long_form", methods=["GET", "POST"])
def long_form():
    if request.method == "POST":
        # Collect input for long form
        inputs = {key: float(request.form[key]) for key in request.form.keys()}  # Use all form keys as features
        features_df = pd.DataFrame([inputs])  # Convert to DataFrame with feature names

        # Predict using the long form model
        prediction = long_form_model.predict(features_df)
        probability = long_form_model.predict_proba(features_df)[0]

        # Map prediction to risk levels
        classes = {0: "Non-Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}
        risk = classes[prediction[0]]

        return render_template(
            "result.html", 
            risk=risk, 
            probability={
                "Non-Diabetes": round(probability[0], 4),
                "Pre-Diabetes": round(probability[1], 4),
                "Diabetes": round(probability[2], 4),
            }, 
            form="Long Form"
        )
    return render_template("long_form.html")

if __name__ == "__main__":
    app.run(debug=True)



