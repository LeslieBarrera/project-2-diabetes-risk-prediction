from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load("optimized_logistic_regression_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse form data
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])
        feature5 = float(request.form['feature5'])

        # Prepare features for prediction
        features = [feature1, feature2, feature3, feature4, feature5]
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)

        # Model accuracy (replace with actual value)
        model_accuracy = 70  # Example: 70%

        # Render result template with prediction
        return render_template(
            'result.html',
            prediction=int(prediction[0]),
            probability=probability[0].tolist(),
            model_accuracy=model_accuracy
        )
    except Exception as e:
        return f"Error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)

