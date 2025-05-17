# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(request.form[x]) for x in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    prediction = model.predict([data])[0]
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)