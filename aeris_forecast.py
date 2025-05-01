import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import time
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# Load model and feature info
model = joblib.load("pm2_5_forecasting_model.pkl")
used_features = joblib.load("used_features.pkl")

# Load city coordinates
city_coords = pd.read_csv("city_coordinates_from_dataset.csv")

def get_city_coordinates(city_name):
    # Normalize and match city_name
    row = city_coords[city_coords["city_name"].str.lower() == city_name.lower()]
    if row.empty:
        raise ValueError(f"City '{city_name}' not found in dataset.")
    return row.iloc[0]["latitude"], row.iloc[0]["longitude"]

def get_real_time_pm25_open_meteo(lat, lon):
    print(f"Fetching real-time PM2.5 data from Open-Meteo for coordinates ({lat}, {lon})...")
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
        f"&hourly=pm2_5&timezone=Asia%2FManila"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    data = response.json()
    timestamps = data.get("hourly", {}).get("time", [])
    pm2_5_values = data.get("hourly", {}).get("pm2_5", [])

    if not timestamps or not pm2_5_values:
        raise Exception("PM2.5 data not available from Open-Meteo.")

    current_time = datetime.now().strftime("%Y-%m-%dT%H:00")
    if current_time in timestamps:
        idx = timestamps.index(current_time)
    else:
        idx = len(pm2_5_values) - 1  # fallback to most recent

    return {
        "pm2_5": pm2_5_values[idx],
        "timestamp": timestamps[idx].replace("T", " ")
    }

def classify_pm2_5(pm2_5):
    if pm2_5 <= 12:
        return "Good"
    elif pm2_5 <= 35.4:
        return "Moderate"
    elif pm2_5 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm2_5 <= 150.4:
        return "Unhealthy"
    elif pm2_5 <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def generate_features(city_name, pm2_5_current, timestamp, lat, lon):
    current_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
    features = {
        "main.aqi": np.nan,
        "components.co": np.nan,
        "components.no": np.nan,
        "components.no2": np.nan,
        "components.o3": np.nan,
        "components.so2": np.nan,
        "components.pm10": np.nan,
        "components.nh3": np.nan,
        "coord.lat": lat,
        "coord.lon": lon,
        "year": current_dt.year,
        "month": current_dt.month,
        "day": current_dt.day,
        "hour": current_dt.hour,
        "day_of_week": current_dt.weekday(),
        "is_weekend": int(current_dt.weekday() >= 5),
        "city_encoded": hash(city_name) % 1000,
        "dayofweek": current_dt.weekday(),
        "season_encoded": 0 if current_dt.month in [11, 12, 1, 2, 3, 4] else 1
    }

    for lag in [1,2,3,4,6,8,12,24,36,48,72]:
        features[f"pm2_5_lag_{lag}h"] = pm2_5_current
    for window in [3, 6]:
        features[f"pm2_5_roll_mean_{window}h"] = pm2_5_current
    features["pm2_5_roll_std_6h"] = 0
    features["pm2_5_roll_median_6h"] = pm2_5_current
    features["pm2_5_roll_min_6h"] = pm2_5_current
    features["pm2_5_roll_max_6h"] = pm2_5_current
    features["pm2_5_city_roll_mean_3h"] = pm2_5_current
    features["pm2_5_city_roll_std_3h"] = 0
    features["pm2_5_city_roll_mean_6h"] = pm2_5_current
    features["pm2_5_city_roll_std_6h"] = 0
    features["lag_1h_x_lag_3h"] = pm2_5_current ** 2
    features["lag_1h_div_lag_3h"] = 1.0

    return pd.DataFrame([features])

def forecast_pm2_5(city_name, days_ahead=3):
    lat, lon = get_city_coordinates(city_name)
    meteo_data = get_real_time_pm25_open_meteo(lat, lon)

    print(f"Real-time PM2.5: {meteo_data['pm2_5']} µg/m³ at {meteo_data['timestamp']}")

    forecast_results = []

    current_pm2_5 = meteo_data["pm2_5"]
    current_time = meteo_data["timestamp"]

    for day in range(days_ahead):
        timestamp = (datetime.strptime(current_time, "%Y-%m-%d %H:%M") + timedelta(days=day)).strftime("%Y-%m-%d %H:%M")
        feature_df = generate_features(city_name, current_pm2_5, timestamp, lat, lon)

        feature_df = feature_df[[col for col in used_features if col in feature_df.columns]]
        feature_df = feature_df.fillna(0)

        scaler = StandardScaler()
        feature_scaled = scaler.fit_transform(feature_df)

        pm2_5_pred = model.predict(feature_scaled)[0]
        category = classify_pm2_5(pm2_5_pred)

        forecast_results.append({
            "date": timestamp.split(" ")[0],
            "predicted_pm2_5": round(pm2_5_pred, 2),
            "category": category
        })

    return pd.DataFrame(forecast_results)

def main():
    city_name = input("Enter the city name (e.g., Manila, Cebu City, etc.): ")
    forecast_df = forecast_pm2_5(city_name, days_ahead=3)
    print("\nAir Quality Forecast:")
    print(forecast_df)

if __name__ == "__main__":
    main()
