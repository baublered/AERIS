import pandas as pd
import requests
import pickle
import joblib
from datetime import datetime, timedelta
from lightgbm import Booster

# Load necessary files
try:
    with open('pm2_5_model_daily.pkl', 'rb') as f:
        model: Booster = pickle.load(f)
    with open('used_features_daily.pkl', 'rb') as f:
        used_features = pickle.load(f)
    city_encoder = joblib.load("city_label_encoder.pkl")
    city_coords = pd.read_csv('city_coordinates_from_dataset.csv')
except FileNotFoundError as e:
    print(f"Error: A required file was not found: {e.filename}")
    print("Please ensure 'pm2_5_model_top10.pkl', 'used_features_top10.pkl', 'city_label_encoder.pkl', and 'city_coordinates_from_dataset.csv' are in the same directory as the script.")
    exit()


# AQI classification thresholds
def classify_pm2_5(value):
    if value is None: # Handle None prediction case
        return "Unknown"
    if value <= 12:
        return "Good"
    elif value <= 35.4:
        return "Moderate"
    elif value <= 55.4:
        return "Poor"
    else: # value > 55.4
        return "Hazardous"


# Time-based features
def create_time_features(df):
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df


# Fetch real-time PM2.5 from Open-Meteo
def fetch_realtime_pm2_5(city_name_input):
    city_column_actual_name = next(
        (col for col in city_coords.columns if col.lower() == "city_name"), None
    )
    if not city_column_actual_name:
        raise ValueError("Critical configuration error: 'city_name' (or similar) column not found in 'city_coordinates_from_dataset.csv'.")

    city_data_row = city_coords.loc[city_coords[city_column_actual_name].str.lower() == city_name_input.lower()]

    if city_data_row.empty:
        raise ValueError(f"City '{city_name_input}' not found in 'city_coordinates_from_dataset.csv'.")

    lat = city_data_row.iloc[0]['latitude']
    lon = city_data_row.iloc[0]['longitude']

    print(f"DEBUG: Using coordinates lat={lat}, lon={lon} for city {city_name_input}")

    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&hourly=pm2_5&timezone=Asia%2FManila&forecast_days=1"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"API request failed for {city_name_input}: {e}")

    if 'hourly' not in data or 'time' not in data['hourly'] or 'pm2_5' not in data['hourly']:
        raise ValueError(f"PM2.5 data not found in API response structure for {city_name_input}.")

    times_raw = data['hourly']['time']
    pm2_5_values_raw = data['hourly']['pm2_5']

    if not times_raw or pm2_5_values_raw is None:
        raise ValueError(f"API returned empty time or PM2.5 data for {city_name_input}.")

    # Filter out None pm2_5 values and ensure datetime conversion
    times, pm2_5_values = [], []
    for i, val_pm2_5 in enumerate(pm2_5_values_raw):
        if val_pm2_5 is not None:
            try:
                times.append(pd.to_datetime(times_raw[i]))
                pm2_5_values.append(float(val_pm2_5))
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping invalid data point from API: time='{times_raw[i]}', pm2_5='{val_pm2_5}'. Error: {e}")
                continue

    if not times: # Check if all data was invalid or filtered out
         raise ValueError(f"API returned PM2.5 data with all null or invalid values for {city_name_input}.")


    df = pd.DataFrame({'datetime': times, 'pm2_5': pm2_5_values})
    return df.sort_values(by='datetime').reset_index(drop=True)


# Forecast function
def forecast_pm2_5(city):
    forecast_results = []

    try:
        df_api_current = fetch_realtime_pm2_5(city)
        if df_api_current.empty:
            return {"error": f"No valid PM2.5 data returned from Open-Meteo API for {city.title()} after filtering."}

        # Get the latest real-time record to start the forecast
        latest_record = df_api_current.iloc[-1]
        latest_datetime_from_api = latest_record['datetime']
        latest_value_from_api = latest_record['pm2_5']

        print(f"\nDEBUG: Initial PM2.5 value fetched from Open-Meteo for {city.title()}:")
        print(f"DEBUG: Timestamp: {latest_datetime_from_api}, PM2.5 from API: {latest_value_from_api:.2f} µg/m³")

        if latest_value_from_api < 1.0:
            print(f"DEBUG: Note: The initial PM2.5 value from the API ({latest_value_from_api:.2f} µg/m³) is very low.")


        # Start the prediction chain with the latest real-time value
        previous_prediction = latest_value_from_api

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An unexpected error occurred during data fetching for {city.title()}: {str(e)}"}

    city_lower = city.lower()
    try:
        correctly_cased_city = next(c for c in city_encoder.classes_ if c.lower() == city_lower)
        encoded_city = city_encoder.transform([correctly_cased_city])[0]
        print(f"DEBUG: Encoded value for city {city}: {encoded_city}")
    except StopIteration:
        known_cities_sample = ", ".join(list(city_encoder.classes_)[:5]) + ("..." if len(list(city_encoder.classes_)) > 5 else "")
        return {"error": f"City '{city}' not found in the pre-trained city encoder. Known cities start with: {known_cities_sample} (matching is case-insensitive but uses stored casing)."}


    current_processing_date = datetime.now().date()

    for i in range(1, 4):
        future_date_to_forecast = current_processing_date + timedelta(days=i)
        # Assuming the daily forecast corresponds to a specific time, let's use midnight for simplicity
        # The model might be trained on a different reference time, which is a potential source of error.
        forecast_datetime_obj = datetime.combine(future_date_to_forecast, datetime.min.time())

        # Prepare the base row dictionary, initializing all used_features to 0.0
        row_dict = {col: 0.0 for col in used_features}

        # Add datetime, city_encoded, and pm2_5 (previous_prediction) which are also used as features
        row_dict['datetime'] = forecast_datetime_obj
        row_dict['city_encoded'] = encoded_city
        row_dict['pm2_5'] = previous_prediction # Using previous prediction as the input PM2.5 value


        # --- Feature Calculation Approximation ---
        # Attempt to make the first forecast day's features more reflective of the real-time value.
        # For subsequent days, approximate using the previous prediction.

        if i == 1:
             # For the first forecast day, use the initial real-time value to approximate recent history
             recent_pm2_5_approx = latest_value_from_api
             # Populate lagged features with the approximation
             if 'pm2_5_lag_1h' in used_features: row_dict['pm2_5_lag_1h'] = recent_pm2_5_approx
             if 'pm2_5_lag_2h' in used_features: row_dict['pm2_5_lag_2h'] = recent_pm2_5_approx
             if 'pm2_5_lag_3h' in used_features: row_dict['pm2_5_lag_3h'] = recent_pm2_5_approx
             if 'pm2_5_lag_4h' in used_features: row_dict['pm2_5_lag_4h'] = recent_pm2_5_approx
             if 'pm2_5_lag_24h' in used_features: row_dict['pm2_5_lag_24h'] = recent_pm2_5_approx # Assuming 24h ago is similar to now

             # Approximate rolling features (simplification: assume constant value)
             if 'pm2_5_roll_mean_3h' in used_features: row_dict['pm2_5_roll_mean_3h'] = recent_pm2_5_approx
             if 'pm2_5_roll_median_6h' in used_features: row_dict['pm2_5_roll_median_6h'] = recent_pm2_5_approx
             if 'pm2_5_roll_min_6h' in used_features: row_dict['pm2_5_roll_min_6h'] = recent_pm2_5_approx
             if 'pm2_5_roll_max_6h' in used_features: row_dict['pm2_5_roll_max_6h'] = recent_pm2_5_approx
             if 'pm2_5_roll_std_6h' in used_features: row_dict['pm2_5_roll_std_6h'] = 0.0 # Std dev would be 0 if constant

        else:
            # For subsequent forecast days (i > 1), use the previous prediction to approximate recent history
            recent_pm2_5_approx = previous_prediction
            if 'pm2_5_lag_24h' in used_features: row_dict['pm2_5_lag_24h'] = previous_prediction # Using previous day's prediction for lag_24h

            # Approximate other lagged features using the previous day's prediction
            if 'pm2_5_lag_1h' in used_features: row_dict['pm2_5_lag_1h'] = recent_pm2_5_approx
            if 'pm2_5_lag_2h' in used_features: row_dict['pm2_5_lag_2h'] = recent_pm2_5_approx
            if 'pm2_5_lag_3h' in used_features: row_dict['pm2_5_lag_3h'] = recent_pm2_5_approx
            if 'pm2_5_lag_4h' in used_features: row_dict['pm2_5_lag_4h'] = recent_pm2_5_approx

            # Approximate rolling features using the previous day's prediction (still a simplification)
            if 'pm2_5_roll_mean_3h' in used_features: row_dict['pm2_5_roll_mean_3h'] = recent_pm2_5_approx
            if 'pm2_5_roll_median_6h' in used_features: row_dict['pm2_5_roll_median_6h'] = recent_pm2_5_approx
            if 'pm2_5_roll_min_6h' in used_features: row_dict['pm2_5_roll_min_6h'] = recent_pm2_5_approx
            if 'pm2_5_roll_max_6h' in used_features: row_dict['pm2_5_roll_max_6h'] = recent_pm2_5_approx
            if 'pm2_5_roll_std_6h' in used_features: row_dict['pm2_5_roll_std_6h'] = 0.0 # Assuming low variation around the predicted value


        row_df = pd.DataFrame([row_dict])
        row_df = create_time_features(row_df)

        try:
            # Ensure column order matches 'used_features'
            row_df_ordered = row_df[used_features]
        except KeyError as e:
            missing_cols = [col for col in used_features if col not in row_df.columns]
            return {"error": f"Critical error: Feature {e} is in 'used_features' but missing from DataFrame. Missing: {missing_cols}. Check 'used_features.pkl' and data preparation."}

        print(f"\nDEBUG: Complete feature input for model for date {future_date_to_forecast.isoformat()} (City: {city.title()}, Input PM2.5 (previous prediction): {previous_prediction:.2f}):")
        print(row_df_ordered.T.to_string(header=False))

        predicted_pm2_5_raw = model.predict(row_df_ordered)[0]

        current_predicted_pm2_5 = float(predicted_pm2_5_raw)
        if current_predicted_pm2_5 < 0:
            print(f"DEBUG: Model raw prediction: {current_predicted_pm2_5:.2f}. Floored to 0.0.")
            current_predicted_pm2_5 = 0.0
        else:
            print(f"DEBUG: Model raw prediction: {current_predicted_pm2_5:.2f}")

        category = classify_pm2_5(current_predicted_pm2_5)

        forecast_results.append({
            "date": future_date_to_forecast.isoformat(),
            "predicted_pm2_5": round(current_predicted_pm2_5, 2),
            "category": category,
        })
        # Update previous_prediction for the next iteration
        previous_prediction = current_predicted_pm2_5

    return forecast_results


# Example usage
if __name__ == "__main__":
    city_input = input("Enter a Philippine city: ").strip()
    if not city_input:
        print("Error: No city entered.")
    else:
        result = forecast_pm2_5(city_input)

        if isinstance(result, dict) and "error" in result:
            print("\nError generating forecast:", result["error"])
        elif not result:
            print(f"\nNo forecast could be generated for {city_input.title()}.")
        else:
            print(f"\nPM2.5 Forecast for {city_input.title()}:")
            for day_forecast in result:
                print(f"{day_forecast['date']}: {day_forecast['predicted_pm2_5']} µg/m³ ({day_forecast['category']})")