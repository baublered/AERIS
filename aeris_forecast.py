import pandas as pd
import numpy as np
import time
import joblib
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta
from urllib.parse import quote_plus

# Load the necessary files
model = joblib.load('pm2_5_forecasting_model.pkl')
used_features = joblib.load('used_features.pkl')
scaler = StandardScaler()

# Constants
API_KEY = "62067aa28e490926060739d6420d490a7ab08c2f"
GEO_URL = "https://api.waqi.info/search/?keyword={city_name}&token={api_key}"
CITY_COORDS_FILE = "city_coordinates_from_dataset.csv"
OUTPUT_FILE = "forecasted_air_quality.csv"

def get_real_time_data(city_name):
    """Fetch real-time air quality data for a given city."""
    city_name_encoded = quote_plus(city_name)
    url = GEO_URL.format(city_name=city_name_encoded, api_key=API_KEY)
    
    try:
        response = requests.get(url)
        data = response.json()

        if data["status"] == "ok" and len(data["data"]) > 0:
            lat = data["data"][0]["station"]["geo"][0]
            lon = data["data"][0]["station"]["geo"][1]
            return lat, lon
        else:
            print(f"❌ No real-time data found for {city_name}.")
            return None, None
    except Exception as e:
        print(f"⚠️ Error fetching real-time data for {city_name}: {e}")
        return None, None

def classify_aqi(pm2_5):
    """Classify the PM2.5 value into AQI categories."""
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

def forecast_pm2_5(city_name, days_ahead=7):
    """Forecast PM2.5 for the given city and the following days."""
    # Load city coordinates from your dataset (you can merge city names and coordinates in advance)
    city_coords_df = pd.read_csv(CITY_COORDS_FILE)
    
    city_coords = city_coords_df[city_coords_df["city_name"] == city_name]
    
    if city_coords.empty:
        print(f"❌ City {city_name} not found in the dataset!")
        return None

    lat, lon = city_coords.iloc[0]["latitude"], city_coords.iloc[0]["longitude"]

    # Fetch real-time data (this will be the input for forecasting)
    lat, lon = get_real_time_data(city_name)
    
    if lat is None or lon is None:
        return None
    
    # Prepare a DataFrame for forecasting the next days
    forecast_dates = [datetime.now() + timedelta(days=i) for i in range(days_ahead)]
    
    forecast_results = []

    for forecast_date in forecast_dates:
        # Prepare features for forecasting
        # Create a DataFrame to match the model's input format
        features = {
            "latitude": [lat],
            "longitude": [lon],
            "year": [forecast_date.year],
            "month": [forecast_date.month],
            "day": [forecast_date.day],
            "hour": [forecast_date.hour],
            "day_of_week": [forecast_date.weekday()],
            "is_weekend": [1 if forecast_date.weekday() >= 5 else 0],
            "main.aqi": [None],  # No AQI value for real-time, will be predicted
            "components.pm2_5": [None],  # Placeholder value for PM2.5, to be predicted
        }

        feature_df = pd.DataFrame(features)

        # Normalize the features using the scaler
        feature_df_scaled = scaler.transform(feature_df[used_features])

        # Make a prediction for PM2.5
        pm2_5_prediction = model.predict(feature_df_scaled)[0]

        # Classify the PM2.5 prediction
        aqi_category = classify_aqi(pm2_5_prediction)

        forecast_results.append({
            "city_name": city_name,
            "date": forecast_date.strftime('%Y-%m-%d'),
            "pm2_5_prediction": pm2_5_prediction,
            "aqi_category": aqi_category
        })

        time.sleep(1)  # To avoid hitting the API limit

    # Save the forecast results to a CSV file
    forecast_df = pd.DataFrame(forecast_results)
    forecast_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Forecast for {city_name} saved to {OUTPUT_FILE}")
    return forecast_df

def main():
    # Accept user input for city name
    city_name = input("Enter the city name (e.g., Manila, Cebu City, etc.): ").strip()

    # Forecast PM2.5 and AQI for the next 7 days
    forecast_df = forecast_pm2_5(city_name, days_ahead=7)

    if forecast_df is not None:
        print(forecast_df)

if __name__ == "__main__":
    main()
