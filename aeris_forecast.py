import pandas as pd
import requests
import pickle
import joblib
from datetime import datetime, timedelta
from lightgbm import Booster
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load necessary files
try:
    with open('pm2_5_model_daily.pkl', 'rb') as f:
        model: Booster = pickle.load(f)
    with open('used_features_daily.pkl', 'rb') as f:
        used_features = pickle.load(f)
    city_encoder = joblib.load("city_label_encoder.pkl")
    city_coords = pd.read_csv('city_coordinates_from_dataset.csv')

# Get the list of cities from the dataset
    cities_list = city_coords['city_name'].tolist()
except FileNotFoundError as e:
    print(f"Error: A required file was not found: {e.filename}")
    print("Please ensure required files are in the same directory as the script.")
    exit()

# AQI classification thresholds
def classify_pm2_5(value):
    if value is None:
        return "Unknown"
    if value <= 12:
        return "Good"
    elif value <= 35.4:
        return "Moderate"
    elif value <= 55.4:
        return "Poor"
    else:
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
        raise ValueError("Critical configuration error: 'city_name' column not found in coordinates file.")

    city_data_row = city_coords.loc[city_coords[city_column_actual_name].str.lower() == city_name_input.lower()]
    if city_data_row.empty:
        raise ValueError(f"City '{city_name_input}' not found in coordinates file.")

    lat = city_data_row.iloc[0]['latitude']
    lon = city_data_row.iloc[0]['longitude']

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
        raise ValueError(f"API returned empty PM2.5 data for {city_name_input}.")

    times, pm2_5_values = [], []
    for i, val_pm2_5 in enumerate(pm2_5_values_raw):
        if val_pm2_5 is not None:
            try:
                times.append(pd.to_datetime(times_raw[i]))
                pm2_5_values.append(float(val_pm2_5))
            except (ValueError, TypeError):
                continue

    if not times:
        raise ValueError(f"All PM2.5 data points were invalid for {city_name_input}.")

    df = pd.DataFrame({'datetime': times, 'pm2_5': pm2_5_values})
    return df.sort_values(by='datetime').reset_index(drop=True)

# Forecast function
def forecast_pm2_5(city):
    forecast_results = []
    actual_values = []  # List to store actual values for performance evaluation
    predicted_values = []  # List to store predicted values
    latest_value_from_api = None  # Define it at the beginning

    try:
        # Fetch the real-time PM2.5 data
        df_api_current = fetch_realtime_pm2_5(city)
        if df_api_current.empty:
            return {"error": f"No valid PM2.5 data returned from API for {city.title()}."}

        # Use daily average instead of single value
        df_today = df_api_current[df_api_current['datetime'].dt.date == datetime.now().date()]
        if df_today.empty:
            return {"error": f"No PM2.5 data for today available from API for {city.title()}."}
        latest_datetime_from_api = df_today['datetime'].max()
        latest_value_from_api = df_today['pm2_5'].mean()

        print(f"Real-time PM2.5 for {city.title()}: {latest_value_from_api:.2f} µg/m³")

        previous_prediction = latest_value_from_api

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

    city_lower = city.lower()
    try:
        correctly_cased_city = next(c for c in city_encoder.classes_ if c.lower() == city_lower)
        encoded_city = city_encoder.transform([correctly_cased_city])[0]
    except StopIteration:
        known_cities_sample = ", ".join(list(city_encoder.classes_)[:5]) + ("..." if len(list(city_encoder.classes_)) > 5 else "")
        return {"error": f"City '{city}' not found in the pre-trained city encoder. Known cities start with: {known_cities_sample}"}

    current_processing_date = datetime.now().date()

    # Add recursive forecasting for 3 days
    for i in range(1, 4):
        future_date = current_processing_date + timedelta(days=i)
        forecast_datetime_obj = datetime.combine(future_date, datetime.min.time())

        row_dict = {col: 0.0 for col in used_features}
        row_dict['datetime'] = forecast_datetime_obj
        row_dict['city_encoded'] = encoded_city
        row_dict['pm2_5'] = previous_prediction

        recent_pm2_5_approx = previous_prediction
        for lag in ['pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h', 'pm2_5_lag_4h', 'pm2_5_lag_24h']:
            if lag in used_features:
                row_dict[lag] = recent_pm2_5_approx
        for roll in ['pm2_5_roll_mean_3h', 'pm2_5_roll_median_6h', 'pm2_5_roll_min_6h', 'pm2_5_roll_max_6h']:
            if roll in used_features:
                row_dict[roll] = recent_pm2_5_approx
        if 'pm2_5_roll_std_6h' in used_features:
            row_dict['pm2_5_roll_std_6h'] = 0.0

        row_df = pd.DataFrame([row_dict])
        row_df = create_time_features(row_df)

        try:
            row_df_ordered = row_df[used_features]
        except KeyError as e:
            missing_cols = [col for col in used_features if col not in row_df.columns]
            return {"error": f"Missing features in input: {missing_cols}"}

        model_prediction = model.predict(row_df_ordered)[0]
        model_prediction = max(0.0, float(model_prediction))
        corrected_prediction = 0.8 * latest_value_from_api + 0.2 * model_prediction
        predicted_pm2_5 = round(max(0.0, corrected_prediction), 2)

        forecast_results.append({
            "date": future_date.isoformat(),
            "predicted_pm2_5": round(predicted_pm2_5, 2),
            "category": classify_pm2_5(predicted_pm2_5),
        })

        # Append actual and predicted values for performance evaluation
        actual_values.append(latest_value_from_api)
        predicted_values.append(predicted_pm2_5)

        previous_prediction = predicted_pm2_5  # Update the previous prediction for the next day's forecast

    # Performance evaluation metrics
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    r2 = r2_score(actual_values, predicted_values)

    return forecast_results, mae, rmse, r2, latest_value_from_api



# Tkinter GUI setup
def show_forecast():
    city_input = city_combobox.get().strip()
    if not city_input:
        messagebox.showerror("Input Error", "Please select a city.")
        return

    result = forecast_pm2_5(city_input)
    if isinstance(result, dict) and "error" in result:
        messagebox.showerror("Error", result["error"])
        return

    forecast_results, mae, rmse, r2, latest_value_from_api = result

    # Display forecast and real-time PM2.5 in the output window
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"Real-time PM2.5 for {city_input.title()}: {latest_value_from_api:.2f} µg/m³\n")
    output_text.insert(tk.END, f"PM2.5 Forecast for {city_input.title()}:\n")
    
    for day in forecast_results:
        output_text.insert(tk.END, f"{day['date']}: {day['predicted_pm2_5']} µg/m³ ({day['category']})\n")

    output_text.insert(tk.END, "\nPerformance Evaluation Metrics:\n")
    output_text.insert(tk.END, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}")

    # Create a plot for the PM2.5 forecast
    dates = [day['date'] for day in forecast_results]
    predicted_pm2_5_values = [day['predicted_pm2_5'] for day in forecast_results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dates, predicted_pm2_5_values, marker='o', color='b', label="Predicted PM2.5")
    ax.set_title(f"PM2.5 Forecast for {city_input.title()}")
    ax.set_xlabel('Date')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.grid(True)
    ax.legend()

    # Embed the plot into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=20)

# Tkinter window
root = tk.Tk()
root.title("PM2.5 Forecast")

tk.Label(root, text="Select a Philippine city:").pack(pady=5)

# Dropdown for city selection
city_combobox = ttk.Combobox(root, values=cities_list, width=40)
city_combobox.pack(pady=5)

forecast_button = tk.Button(root, text="Get Forecast", command=show_forecast)
forecast_button.pack(pady=10)

output_text = tk.Text(root, height=15, width=50)
output_text.pack(pady=10)

root.mainloop()