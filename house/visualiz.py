from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__,template_folder='transplant')

# Load data
df = pd.read_csv('C:/Users/bsrih/Desktop/house/dataset/vietnamhousds.csv')

@app.route('/')
def home():
    # Create a plot: Area vs. Price
    plt.figure(figsize=(8,5))
    plt.scatter(df['Area'], df['Price'], color='blue', alpha=0.6)
    plt.title('House Area vs. Price')
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price ($)')
    plt.grid(True)

    # Save the plot as a static image
    img_path = os.path.join('static', 'plot.png')
    plt.savefig(img_path)
    plt.close()

    return render_template('visualiz.html', plot_url=img_path)

if __name__ == '__main__':
    app.run(debug=True)
