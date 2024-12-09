
# Project: Diabetes Prediction Model

## Overview

This project focuses on predicting diabetes using machine learning models. The analysis emphasizes feature selection, model evaluation, and the deployment of a simplified and efficient predictive system.

## Feature Selection

### Insights

#### Top Features
- **BMI, Age, Income**: These features significantly reduce impurity, indicating their strong relationship with diabetes.

#### Moderately Important Features
- **PhysHlth, Education, GenHlth**: These contribute meaningfully but are less dominant than the top features.

#### Low-Importance Features
- **CholCheck, AnyHealthcare, HvyAlcoholConsump**: These provide minimal predictive value.

### Final Selection
- **Selected Features**: BMI, Age, Income, PhysHlth, and Education.
- **Purpose**: Reduces dimensionality and improves model efficiency.

## Model Selection

### Recommendation
- **Best Model**: Logistic Regression

### Key Metrics
- **Recall**: 0.7
- **F1-Score**: 0.39
- **Accuracy**: 0.7

### Reasoning
The model prioritizes recall, which is critical in healthcare to minimize missed diabetes cases.

### Alternative
- **Random Forest**: Offers a balance between precision and recall but has a lower recall score compared to Logistic Regression.

## Short Form Deployment

### Simplified Prediction Model
- **Features**: BMI, Age, Income, PhysHlth, and Education.

### Benefits
- Reduces input complexity.
- Enhances user-friendliness and accessibility.
- Maintains predictive power.

## Deployment Recommendations

### Logging
- Record inputs, predictions, and errors for debugging and monitoring.

### Model Maintenance
- **Retraining**: Regularly update the model with new data.
- **Validation**: Implement stricter input validation (e.g., BMI between 12–98).

### Optimization
- Continuously monitor and refine the model to ensure performance.

## Acknowledgments

This project highlights the critical importance of feature engineering, model selection, and user-focused deployment in creating impactful machine learning solutions.
