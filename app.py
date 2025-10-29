"""
Iris Flower Classification Web Application
Flask app for predicting Iris flower species
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ model.pkl not found! Run train_model.py first.")
    exit(1)

# Species mapping
species_mapping = {
    0: "Iris-Setosa",
    1: "Iris-Versicolor",
    2: "Iris-Virginica"
}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    input_values = None
    confidence = None
    
    if request.method == 'POST':
        try:
            # Get input values from form
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            
            # Store input values for display
            input_values = {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }
            
            # Create input array
            data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Make prediction
            pred_class = model.predict(data)[0]
            prediction = species_mapping[pred_class]
            
            # Get prediction probability (confidence)
            probabilities = model.predict_proba(data)[0]
            confidence = max(probabilities) * 100
            
        except ValueError:
            prediction = "Invalid input! Please enter numeric values."
    
    return render_template('index.html', 
                         prediction=prediction, 
                         input_values=input_values,
                         confidence=confidence)

if __name__ == "__main__":
    app.run(debug=False)
