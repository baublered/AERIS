import pandas as pd
import numpy as np
import joblib
import os

# === Load Model and Features with Error Handling ===
model_path = "pm2_5_forecasting_model.pkl"
features_path = "used_features.pkl"

if not os.path.exists(model_path) or not os.path.exists(features_path):
    print("❌ Required files not found. Please ensure the following files exist:")
    if not os.path.exists(model_path):
        print(f"   - Missing: {model_path}")
    if not os.path.exists(features_path):
        print(f"   - Missing: {features_path}")
    exit()

model = joblib.load(model_path)
features = joblib.load(features_path)

# === Load and preprocess the data ===
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

# === User Input and Validation ===
city_input = input("🏙️ Enter a city name to forecast PM2.5: ").strip()
city_df = df[df["city_name"].str.lower() == city_input.lower()].sort_values("datetime", ascending=False)

if city_df.empty:
    available_cities = sorted(df["city_name"].unique())
    print(f"❌ City '{city_input}' not found in the dataset.")
    print("📍 Available cities include:")
    print(", ".join(available_cities[:10]) + "...")
    print(f"👉 Total cities available: {len(available_cities)}")
    exit()

# === Single-Day Forecast ===
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
    category = "Unhealthy for Sensitive Groups"
elif y_pred_pm25 <= 150.4:
    category = "Unhealthy"
elif y_pred_pm25 <= 250.4:
    category = "Very Unhealthy"
else:
    category = "Hazardous"

# Output the result
print(f"\n🔮 Forecasted PM2.5 for {city_input.title()}: {y_pred_pm25:.2f} µg/m³")
print(f"🛡️ Air Quality Category: {category}")
print(f"📅 Date and Time of Forecast: {latest_record['datetime']}")