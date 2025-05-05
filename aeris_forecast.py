import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from feature_engineering import create_time_features, create_interaction_features, create_city_rolling_features, create_lag_features
from data_preprocessing import get_city_coordinates
import requests

# Load the city encoder to transform city name
city_encoder = joblib.load('city_label_encoder.pkl')  

def fetch_open_meteo_data(lat, lon):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        pm2_5_values = data['hourly']['pm2_5']
        timestamps = data['hourly']['time']

        df = pd.DataFrame({
            "datetime": pd.to_datetime(timestamps),
            "components.pm2_5": pm2_5_values
        })
        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"Error fetching data from Open-Meteo: {e}")
        return None

def classify_aqi(pm2_5):
    if pm2_5 <= 12:
        return "Good"
    elif pm2_5 <= 35.4:
        return "Moderate"
    elif pm2_5 <= 55.4:
        return "Poor"
    else:
        return "Hazardous"

def prepare_forecast_row(base_df, forecast_date, city_name, lat, lon, pm25_value):
    """Prepare a forecast row with all required features"""
    forecast_dt = pd.to_datetime(forecast_date)
    
    base_row = {
        "datetime": forecast_dt,
        "city_name": city_name,
        "lat": lat,
        "lon": lon,
        "season": 'dry' if forecast_dt.month in [1, 2, 3, 4, 11, 12] else 'wet',
        "components.pm2_5": pm25_value
    }
    
    # Add all missing features
    missing_features = ['main.aqi', 'components.co', 'components.no', 'components.no2', 
                    'components.o3', 'components.so2', 'components.pm10', 'components.nh3', 
                    'coord.lon', 'coord.lat', 'year', 'day', 'day_of_week', 'city_encoded']
    
    for feature in missing_features:
        if feature not in base_row:
            if feature == 'city_encoded':
                base_row[feature] = city_encoder.transform([city_name])[0]
            elif feature in ['coord.lon', 'coord.lat']:
                base_row[feature] = lon if feature == 'coord.lon' else lat
            elif feature == 'year':
                base_row[feature] = forecast_dt.year
            elif feature == 'day':
                base_row[feature] = forecast_dt.day
            elif feature == 'day_of_week':
                base_row[feature] = forecast_dt.weekday()
            else:
                base_row[feature] = 0
    
    # Convert to DataFrame and concatenate
    forecast_row = pd.DataFrame([base_row])
    full_df = pd.concat([base_df, forecast_row], ignore_index=True)
    
    # Feature engineering
    full_df = create_time_features(full_df)
    full_df = create_city_rolling_features(full_df)
    full_df = create_lag_features(full_df)
    full_df = create_interaction_features(full_df)
    
    # Return only the forecast row with all features
    forecast_row = full_df[full_df['datetime'] == forecast_dt].copy()
    return forecast_row

if __name__ == "__main__":
    city_name = input("Enter the city name (e.g., Manila, Cebu City, etc.): ").strip()

    # === Step 1: Load model and features ===
    model = joblib.load("pm2_5_forecasting_model.pkl")
    used_features = joblib.load("used_features.pkl")

    # === Step 2: Get city coordinates ===
    try:
        coord_df = pd.read_csv("city_coordinates_from_dataset.csv")
        lat, lon = get_city_coordinates(city_name, coord_df)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)

    # === Step 3: Get recent PM2.5 data ===
    print(f"Fetching recent PM2.5 data from Open-Meteo for coordinates ({lat}, {lon})...")
    pm25_df = fetch_open_meteo_data(lat, lon)

    if pm25_df is None or pm25_df.empty:
        print("❌ Could not fetch recent PM2.5 data.")
        exit(1)

    print(f"Found {len(pm25_df)} data points from Open-Meteo")
    print("Latest data points:")
    print(pm25_df.tail(3))

    # === Step 4: Get latest PM2.5 value ===
    today = datetime.today().date()
    latest_entry = pm25_df.iloc[-1]
    real_pm25 = latest_entry['components.pm2_5']
    print(f"\nReal-time PM2.5: {real_pm25} µg/m³ at {today}")

    # === Step 5: Prepare historical data ===
    pm25_df['city_name'] = city_name
    pm25_df['lat'] = lat
    pm25_df['lon'] = lon
    pm25_df['season'] = pm25_df['datetime'].dt.month.apply(lambda m: 'dry' if m in [1, 2, 3, 4, 11, 12] else 'wet')

    # === Step 6: Feature Engineering for historical data ===
    print("\nPerforming feature engineering on historical data...")
    feature_df = create_time_features(pm25_df.copy())
    feature_df = create_city_rolling_features(feature_df)
    feature_df = create_lag_features(feature_df)
    feature_df = create_interaction_features(feature_df)
    feature_df.dropna(inplace=True)
    print(f"Feature engineering complete. Final shape: {feature_df.shape}")

    # === Step 7: Forecast next 3 days ===
    forecast_dates = [today + timedelta(days=1), today + timedelta(days=2), today + timedelta(days=3)]
    forecast = []
    
    for forecast_date in forecast_dates:
        print(f"\nPreparing forecast for {forecast_date}...")
        
        try:
            # Prepare forecast row with all features
            forecast_row = prepare_forecast_row(
                feature_df.copy(), 
                forecast_date, 
                city_name, 
                lat, 
                lon, 
                real_pm25
            )
            
            if forecast_row.empty:
                raise ValueError("Feature engineering failed for forecast date")
                
            # Make prediction
            predict_df = forecast_row[used_features]
            pred = model.predict(predict_df.values)
            pred_inv = np.expm1(pred)[0]
            category = classify_aqi(pred_inv)
            
            forecast.append({
                "date": forecast_date,
                "predicted_pm2_5": round(pred_inv, 2),
                "category": category
            })
            
            print(f"Forecast: {round(pred_inv, 2)} µg/m³ ({category})")
            
        except Exception as e:
            print(f"Warning: {e}")
            forecast.append({
                "date": forecast_date,
                "predicted_pm2_5": None,
                "category": "Insufficient data"
            })

    print("\nAir Quality Forecast:")
    print(pd.DataFrame(forecast))