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
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from collections import defaultdict
import concurrent.futures
import time


# Load necessary files
try:
    with open('aqi_forecasting_model.pkl', 'rb') as f:
        model: Booster = pickle.load(f)
    with open('used_features.pkl', 'rb') as f:
        used_features = pickle.load(f)
    city_encoder = joblib.load("city_label_encoder.pkl")
    city_coords = pd.read_csv('city_coordinates_from_dataset.csv')
    cities_list = city_coords['city_name'].tolist()
except FileNotFoundError as e:
    print(f"Error: A required file was not found: {e.filename}")
    print("Please ensure required files ('aqi_forecasting_model.pkl', 'used_features.pkl', 'city_label_encoder.pkl', 'city_coordinates_from_dataset.csv') are in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred during initial file loading: {e}")
    exit()


def classify_aqi_openweathermap(value):
    """
    Classifies AQI based on OpenWeatherMap's 1-5 scale.
    1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor.
    """
    if value is None: return "Unknown"
    val_int = round(float(value))
    if val_int == 1: return "Good"
    elif val_int == 2: return "Fair"
    elif val_int == 3: return "Moderate"
    elif val_int == 4: return "Poor"
    elif val_int >= 5: return "Very Poor"
    else: return "Unknown"


def create_time_features(df):
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    return df


def forecast_for_day(i, current_processing_date, current_aqi_lag1, current_aqi_lag2, used_features, model, city, encoded_city):
    # Function to forecast AQI for a single day
    future_date = current_processing_date + timedelta(days=i)
    forecast_datetime_obj = datetime.combine(future_date, datetime.min.time())
    row_dict = {col: 0.0 for col in used_features}
    row_dict['datetime'] = forecast_datetime_obj

    if 'aqi_lag1' in used_features: row_dict['aqi_lag1'] = current_aqi_lag1
    if 'aqi_lag2' in used_features: row_dict['aqi_lag2'] = current_aqi_lag2
    if 'city_name' in used_features: row_dict['city_name'] = encoded_city
    row_df = pd.DataFrame([row_dict])
    row_df = create_time_features(row_df)
    row_df_ordered = row_df[used_features]
    model_prediction_raw = model.predict(row_df_ordered)[0]
    predicted_aqi_value = round(float(model_prediction_raw))

    return {
        "date": future_date.isoformat(),
        "predicted_aqi": predicted_aqi_value,
        "category": classify_aqi_openweathermap(predicted_aqi_value),
    }


def forecast_aqi_logic_parallel(city):
    forecast_results_data = []
    actual_values_for_metrics = []
    predicted_values_for_metrics = []

    current_aqi_value = fetch_realtime_air_data(city)
    latest_aqi_from_api = float(current_aqi_value)
    current_aqi_lag1 = latest_aqi_from_api
    current_aqi_lag2 = latest_aqi_from_api
    city_lower = city.lower()
    correctly_cased_city = next(c for c in city_encoder.classes_ if c.lower() == city_lower)
    encoded_city = city_encoder.transform([correctly_cased_city])[0]
    current_processing_date = datetime.now().date()

    # Parallelizing the forecast for 3 days
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_day = {
            executor.submit(forecast_for_day, i, current_processing_date, current_aqi_lag1, current_aqi_lag2, used_features, model, city, encoded_city): i
            for i in range(1, 4)
        }
        for future in concurrent.futures.as_completed(future_to_day):
            result = future.result()
            forecast_results_data.append(result)
            predicted_values_for_metrics.append(result['predicted_aqi'])
            actual_values_for_metrics.append(latest_aqi_from_api)

    # Calculate metrics
    mae, rmse, r2 = None, None, None
    try:
        if actual_values_for_metrics and predicted_values_for_metrics:
            mae = mean_absolute_error(actual_values_for_metrics, predicted_values_for_metrics)
            rmse = np.sqrt(mean_squared_error(actual_values_for_metrics, predicted_values_for_metrics))
            if np.var(actual_values_for_metrics) == 0: r2 = float('nan')
            else: r2 = r2_score(actual_values_for_metrics, predicted_values_for_metrics)
    except Exception as e:
        print(f"Warning: Error calculating performance metrics: {str(e)}")
    
    return forecast_results_data, mae, rmse, r2, latest_aqi_from_api


def forecast_aqi_logic_sequential(city):
    forecast_results_data = []
    actual_values_for_metrics = []
    predicted_values_for_metrics = []

    current_aqi_value = fetch_realtime_air_data(city)
    latest_aqi_from_api = float(current_aqi_value)
    current_aqi_lag1 = latest_aqi_from_api
    current_aqi_lag2 = latest_aqi_from_api
    city_lower = city.lower()
    correctly_cased_city = next(c for c in city_encoder.classes_ if c.lower() == city_lower)
    encoded_city = city_encoder.transform([correctly_cased_city])[0]
    current_processing_date = datetime.now().date()

    for i in range(1, 4):
        future_date = current_processing_date + timedelta(days=i)
        forecast_datetime_obj = datetime.combine(future_date, datetime.min.time())
        row_dict = {col: 0.0 for col in used_features}
        row_dict['datetime'] = forecast_datetime_obj

        if 'aqi_lag1' in used_features: row_dict['aqi_lag1'] = current_aqi_lag1
        if 'aqi_lag2' in used_features: row_dict['aqi_lag2'] = current_aqi_lag2
        if 'city_name' in used_features: row_dict['city_name'] = encoded_city
        row_df = pd.DataFrame([row_dict])
        row_df = create_time_features(row_df)
        row_df_ordered = row_df[used_features]
        model_prediction_raw = model.predict(row_df_ordered)[0]
        predicted_aqi_value = round(float(model_prediction_raw))

        forecast_results_data.append({
            "date": future_date.isoformat(),
            "predicted_aqi": predicted_aqi_value,
            "category": classify_aqi_openweathermap(predicted_aqi_value),
        })
        predicted_values_for_metrics.append(predicted_aqi_value)
        actual_values_for_metrics.append(latest_aqi_from_api)

        current_aqi_lag2 = current_aqi_lag1
        current_aqi_lag1 = predicted_aqi_value

    mae, rmse, r2 = None, None, None
    try:
        if actual_values_for_metrics and predicted_values_for_metrics:
            mae = mean_absolute_error(actual_values_for_metrics, predicted_values_for_metrics)
            rmse = np.sqrt(mean_squared_error(actual_values_for_metrics, predicted_values_for_metrics))
            if np.var(actual_values_for_metrics) == 0: r2 = float('nan')
            else: r2 = r2_score(actual_values_for_metrics, predicted_values_for_metrics)
    except Exception as e:
        print(f"Warning: Error calculating performance metrics: {str(e)}")
    
    return forecast_results_data, mae, rmse, r2, latest_aqi_from_api


def compare_parallel_vs_sequential(city):
    start_time = time.time()
    forecast_aqi_logic_sequential(city)
    sequential_time = time.time() - start_time

    start_time = time.time()
    forecast_aqi_logic_parallel(city)
    parallel_time = time.time() - start_time

    print(f"Sequential Time: {sequential_time:.4f} seconds")
    print(f"Parallel Time: {parallel_time:.4f} seconds")
    print(f"Time saved: {sequential_time - parallel_time:.4f} seconds")


# Tkinter GUI Setup
root = tk.Tk()
root.title("AERIS - AQI Forecast (OpenWeatherMap 1-5 Scale)")

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

control_frame = ttk.Frame(main_frame)
control_frame.pack(pady=5)

city_label = tk.Label(control_frame, text="Select City:")
city_label.pack(side=tk.LEFT, padx=5)
city_combobox = ttk.Combobox(control_frame, values=cities_list, width=30)
if cities_list: city_combobox.current(0)
city_combobox.pack(side=tk.LEFT, padx=5)

forecast_button = tk.Button(control_frame, text="Get AQI Forecast (1-5)", command=lambda: show_forecast())
forecast_button.pack(side=tk.LEFT, padx=5)
clear_button = tk.Button(control_frame, text="Clear", command=lambda: clear_results())
clear_button.pack(side=tk.LEFT, padx=5)

results_frame = ttk.Frame(main_frame)
results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

text_frame = ttk.LabelFrame(results_frame, text="AQI Forecast Details (1-5 Scale)", padding="10")
text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, anchor='n')
output_text = tk.Text(text_frame, height=20, width=55)
output_text.pack(fill=tk.BOTH, expand=True)

plot_container_frame = ttk.Frame(results_frame)
plot_container_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
plot_frame = tk.Frame(plot_container_frame)
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
plot_performance_frame = tk.Frame(plot_container_frame)
plot_performance_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10)

root.mainloop()
