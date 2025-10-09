def create_features(df):
    """Add or transform features for better prediction."""
    # Example: combine temperature and humidity into a comfort index
    if 'Temperature' in df.columns and 'Humidity' in df.columns:
        df['Comfort_Index'] = (df['Temperature'] * (100 - df['Humidity'])) / 100
        print("✅ Feature 'Comfort_Index' created.")
    return df
