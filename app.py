from flask import Flask, render_template

import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    # Load and clean data
    df = pd.read_csv('c:/Users/bsrih/Desktop/brain_stroke/netflix.csv')
    df.dropna(subset=['type', 'country', 'release_year', 'listed_in'], inplace=True)

    # Add analytics
    genre_counts = df['listed_in'].str.split(', ').explode().value_counts().head(5)
    country_counts = df['country'].value_counts().head(5)
    release_counts = df['release_year'].value_counts().sort_index()
    type_counts = df['type'].value_counts()

    # Pass data to template
    return render_template('index.html',
                           genres=genre_counts.index.tolist(),
                           genre_values=genre_counts.tolist(),
                           countries=country_counts.index.tolist(),
                           country_values=country_counts.tolist(),
                           years=release_counts.index.tolist(),
                           release_values=release_counts.tolist(),
                           types=type_counts.index.tolist(),
                           type_values=type_counts.tolist()
                           )

if __name__ == '__main__':
    app.run(debug=True)
