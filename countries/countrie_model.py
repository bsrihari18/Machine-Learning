import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
from math import sqrt

# Load the dataset
df = pd.read_csv("dataset/population_growth.csv")

# Drop rows with missing 'Growth Rate (%)'
df = df.dropna(subset=["Growth Rate (%)"])

# Drop non-numeric columns that can't be used directly
X = df.drop(columns=["Country", "ISO3", "Decade", "Growth Rate (%)"])
y = df["Growth Rate (%)"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Model trained. RMSE: {rmse:.3f}")

# Save the model to file
with open("population_growth_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("📦 Model saved as 'population_growth_model.pkl'")
