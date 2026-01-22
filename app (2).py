from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Use absolute paths so Render/Python doesn't get confused
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'model', 'breast_cancer_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'model', 'scaler.pkl'))
le = joblib.load(os.path.join(BASE_DIR, 'model', 'label_encoder.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the 5 numeric inputs from the HTML form
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        
        # Scale the data (Mandatory for SVM)
        scaled_features = scaler.transform(final_features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Convert 0/1 back to 'Benign' or 'Malignant'
        result = le.inverse_transform(prediction)[0]
        display_text = "Malignant (High Risk)" if result == 'M' else "Benign (Low Risk)"
        
        return render_template('index.html', prediction_text=f'Result: {display_text}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)