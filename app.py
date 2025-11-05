from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and columns
model = joblib.load('heart_disease_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.form.to_dict()
    
    # Convert height and weight to float, calculate BMI
    try:
        height_cm = float(data['Height'])
        height_m = height_cm / 100
        weight = float(data['Weight'])
        data['BMI'] = weight / (height_m * height_m)
    except (ValueError, ZeroDivisionError):
        data['BMI'] = 0

    # Create a pandas DataFrame from the input data
    query = pd.DataFrame([data])
    
    # Align the query DataFrame with the model's training columns
    query = query.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction_proba = model.predict_proba(query)[:, 1]
    confidence_score = round(prediction_proba[0] * 100, 2)

    # Rule-based adjustments
    if data.get('Cholesterol') == 'Yes':
        confidence_score += 20
    if data.get('Smoking') == 'Yes':
        confidence_score += 15
    if data.get('AlcoholDrinking') == 'Yes':
        confidence_score += 10
    if data.get('Diabetic') == 'Yes':
        confidence_score += 15
        
    # Cap the score at 99%
    confidence_score = min(confidence_score, 99.0)

    # Determine the reason for the prediction
    reason = "Multiple factors"
    if confidence_score > 50:
        # Get feature importances
        importances = model.feature_importances_
        feature_names = model_columns
        
        # Create a dictionary of feature importances
        feature_importance_dict = dict(zip(feature_names, importances))
        
        # Get the top 3 features for this prediction
        query_dict = query.to_dict('records')[0]
        
        # Filter for features that are present in the user's input and have high importance
        user_features = {k: v for k, v in query_dict.items() if v > 0}
        
        if user_features:
            # Sort user features by importance
            sorted_user_features = sorted(user_features.keys(), key=lambda x: feature_importance_dict.get(x, 0), reverse=True)
            
            # Get the top reason
            top_feature = sorted_user_features[0]
            reason = f"High risk primarily due to {top_feature.replace('_', ' ')}"
        else:
            reason = "High risk due to a combination of factors."

    else:
        reason = "Low risk based on the provided information"

    return render_template('result.html', prediction=confidence_score, reason=reason)

if __name__ == '__main__':
    app.run(debug=True)
