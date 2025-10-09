from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def evaluate_model(df):
    model = joblib.load('models/aqi_model.pkl')
    X = df.drop('AQI', axis=1)
    y_true = df['AQI']
    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"📈 Model Evaluation: MAE={mae:.2f}, R²={r2:.2f}")
