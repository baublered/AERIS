import pandas as pd
import numpy as np
import joblib

# Load the trained model and used features
model = joblib.load("pm2_5_forecasting_model.pkl")
features = joblib.load("used_features.pkl")

# Load and preprocess the data
from data_preprocessing import load_and_clean_data
from feature_engineering import (
    create_lag_features,
    create_time_features,
    create_city_rolling_features,
    create_interaction_features
)

print("📦 Loading and preparing data...")
df = load_and_clean_data("cleaned_air_quality_data.csv")

# Make sure datetime is parsed
df["datetime"] = pd.to_datetime(df["datetime"])

# Feature engineering
df = create_lag_features(df)
df = create_time_features(df)
df = create_city_rolling_features(df)
df = create_interaction_features(df)

# Drop any NA rows after feature engineering
df.dropna(inplace=True)

# === Simple User Interface ===
city_input = input("🏙️ Enter a city name to forecast PM2.5: ").strip()

# Filter the latest data for that city
city_df = df[df["city_name"].str.lower() == city_input.lower()].sort_values("datetime", ascending=False)

if city_df.empty:
    print(f"❌ City '{city_input}' not found in dataset.")
else:
    # Take the most recent record
    latest_record = city_df.iloc[0]

    # Prepare feature input
    X_latest = latest_record[features].values.reshape(1, -1)

    # Predict
    y_pred_log = model.predict(X_latest, num_iteration=model.best_iteration)
    y_pred_pm25 = np.expm1(y_pred_log)[0]  # invert log1p

    # Classify the air quality
    if y_pred_pm25 <= 12:
        category = "Good"
    elif y_pred_pm25 <= 35.4:
        category = "Moderate"
    elif y_pred_pm25 <= 55.4:
        category = "Poor"
    else:
        category = "Hazardous"

    # Output the result
    print(f"\n🔮 Forecasted PM2.5 for {city_input.title()}: {y_pred_pm25:.2f} µg/m³")
    print(f"🛡️ Air Quality Category: {category}")
    print(f"📅 Date and Time of Forecast: {latest_record['datetime']}")