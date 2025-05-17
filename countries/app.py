from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("population_growth_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            year = float(request.form['year'])
            population = float(request.form['population'])
            pop_growth = float(request.form['pop_growth'])

            # Predict
            input_data = np.array([[year, population, pop_growth]])
            pred = model.predict(input_data)[0]
            prediction = round(pred, 3)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
