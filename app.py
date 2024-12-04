from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the saved model
short_form_model = load("optimized_logistic_regression_model.joblib")
long_form_model = load("weighted_xgboost_model.joblib")

@app.route('/')
def home():
    return render_template('short_form.html')

@app.route('/form/<form_type>')
def form(form_type):
    if form_type == "short":
        return render_template('short_form.html')
    elif form_type == "long":
        return render_template('long_form.html')
    else:
        return "Invalid form type.", 400

@app.route('/predict', methods=['POST'])
def predict():
    # Check if form data is present
    if request.form:
        try:
            features = [float(value) for value in request.form.getlist('features[]')]
            form_type = request.form.get('form_type')
        except ValueError:
            return "Invalid input: ensure all fields are properly filled.", 400
    else:
        return "Unsupported Media Type. Use the form to submit your data.", 415

    # Choose model based on form type
    model = short_form_model if form_type == "short" else long_form_model

    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)

    # Predict and get probabilities
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array).max()

    # Render results
    result = "Low Risk" if prediction == 0 else "High Risk"
    return render_template(
        'result.html',
        prediction=result,
        probability=f"{probability * 100:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
