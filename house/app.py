from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__,template_folder="transplant")

# Load the trained model
with open('house_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    features = [[area, bedrooms]]
    prediction = model.predict(features)[0]
    return render_template('index.html', prediction_text=f'Predicted Price: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
