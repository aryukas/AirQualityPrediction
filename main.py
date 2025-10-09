from src.data_preprocessing import load_and_preprocess
from src.model_training import train_model
import os
import joblib

data_path = "data/air_quality_data.csv"
model_path = "models/air_quality_model.pkl"

# Load data
X, y = load_and_preprocess(data_path)

# Train or load model
if os.path.exists(model_path):
    print("✅ Loading existing model...")
    model = joblib.load(model_path)
else:
    print("📊 Training new model...")
    model = train_model(X, y, save_path=model_path)

print("✅ Model ready for predictions!")
