import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("C:/Users/bsrih/Desktop/house/dataset/vietnamhousds.csv")

df = df.dropna()

X = df[['Area','Bedrooms']]
y = df['Price']

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

with open('house_model.pkl', 'wb') as f:
    pickle.dump(model,f)

print("Model trained and saved successfully as house_model.pkl")

