import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

# 1️⃣ Load CSV
csv_path = os.path.join("data", "air_quality_data.csv")
df = pd.read_csv(csv_path)
print("✅ CSV loaded successfully!")
print(df.head())

# 2️⃣ Use only numeric features
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']]
y = df['AQI']

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train a tiny Random Forest model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# 5️⃣ Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"📊 MAE on test set: {mae:.2f}")

# 6️⃣ Save model
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "test_model.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved at {model_path}")
