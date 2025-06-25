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


# Load necessary files
try:
    # AQI Model (Required)
    with open('aqi_forecasting_model.pkl', 'rb') as f: 
        aqi_model: Booster = pickle.load(f)
    with open('used_features.pkl', 'rb') as f:
        aqi_used_features = pickle.load(f)

    city_encoder = joblib.load("city_label_encoder.pkl")
    city_coords = pd.read_csv('city_coordinates_from_dataset.csv')
    cities_list = city_coords['city_name'].tolist()

    # PM2.5 Model (Optional)
    pm25_model = None
    pm25_used_features = None
    try:
        with open('pm25_forecasting_model.pkl', 'rb') as f:
            pm25_model: Booster = pickle.load(f)
        with open('pm25_used_features.pkl', 'rb') as f:
            pm25_used_features = pickle.load(f)
        print("Successfully loaded PM2.5 prediction model. Forecasts will be dynamic.")
    except FileNotFoundError:
        print("Warning: PM2.5 model files not found. Falling back to simpler AQI forecast.")

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
    df['year'] = df['datetime'].dt.year
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    return df

OPENWEATHERMAP_API_KEY = "046db00705c9a85bd3daa1f61ea04d4a"
OPENWEATHERMAP_API_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

def fetch_realtime_air_data(city_name_input):
    city_column_actual_name = next((col for col in city_coords.columns if col.lower() == "city_name"), None)
    if not city_column_actual_name:
        raise ValueError("Critical: 'city_name' column not found in coordinates file.")
    city_data_row = city_coords.loc[city_coords[city_column_actual_name].str.lower() == city_name_input.lower()]
    if city_data_row.empty:
        raise ValueError(f"City '{city_name_input}' not found in coordinates file.")
    lat = city_data_row.iloc[0]['latitude']
    lon = city_data_row.iloc[0]['longitude']
    url = f"{OPENWEATHERMAP_API_URL}?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"API request failed for {city_name_input}: {e}")
    if 'list' not in data or not data['list']:
        raise ValueError(f"Unexpected API response format for {city_name_input}: 'list' missing or empty.")
    return data['list'][0]

def forecast_aqi_logic(city):
    forecast_results_data = []
    
    # Lists for AQI performance metrics
    actual_aqi_for_metrics = []
    predicted_aqi_for_metrics = []
    
    # Lists for PM2.5 performance metrics
    actual_pm25_for_metrics = []
    predicted_pm25_for_metrics = []

    city_data_row = city_coords.loc[city_coords['city_name'].str.lower() == city.lower()]
    if city_data_row.empty:
        return {"error": f"City '{city}' not found in coordinates file."}
    lat = city_data_row.iloc[0]['latitude']
    lon = city_data_row.iloc[0]['longitude']

    try:
        realtime_air_data = fetch_realtime_air_data(city)
        current_aqi_value = realtime_air_data['main']['aqi']
        current_components = realtime_air_data.get('components', {})
        
        latest_aqi_from_api = float(current_aqi_value)
        latest_pm25_from_api = current_components.get('pm2_5', 0.0)

        current_aqi_lag1 = latest_aqi_from_api
        current_aqi_lag2 = latest_aqi_from_api

        current_pm25_lag1 = latest_pm25_from_api
        current_pm25_lag2 = latest_pm25_from_api

    except (ValueError, KeyError) as e:
        return {"error": f"Error fetching or parsing real-time data: {str(e)}"}

    try:
        correctly_cased_city = next(c for c in city_encoder.classes_ if c.lower() == city.lower())
        encoded_city = city_encoder.transform([correctly_cased_city])[0]
    except StopIteration:
        return {"error": f"City '{city}' not recognized by city encoder."}

    current_processing_date = datetime.now().date()

    for i in range(1, 4):
        future_date = current_processing_date + timedelta(days=i)
        forecast_datetime_obj = datetime.combine(future_date, datetime.min.time())
        
        predicted_pm25 = None
        if pm25_model and pm25_used_features:
            pm25_row_dict = {col: 0.0 for col in pm25_used_features}
            pm25_row_dict['datetime'] = forecast_datetime_obj
            if 'pm25_lag1' in pm25_used_features: pm25_row_dict['pm25_lag1'] = current_pm25_lag1
            if 'pm25_lag2' in pm25_used_features: pm25_row_dict['pm25_lag2'] = current_pm25_lag2
            if 'coord.lon' in pm25_used_features: pm25_row_dict['coord.lon'] = lon
            if 'coord.lat' in pm25_used_features: pm25_row_dict['coord.lat'] = lat
            if 'city_encoded' in pm25_used_features: pm25_row_dict['city_encoded'] = encoded_city

            pm25_df = pd.DataFrame([pm25_row_dict])
            pm25_df = create_time_features(pm25_df)
            pm25_df_ordered = pm25_df[pm25_used_features]
            
            predicted_pm25 = pm25_model.predict(pm25_df_ordered)[0]
            predicted_pm25 = max(0, predicted_pm25)
        
        aqi_row_dict = {col: 0.0 for col in aqi_used_features}
        aqi_row_dict['datetime'] = forecast_datetime_obj
        if 'components.pm2_5' in aqi_used_features:
            aqi_row_dict['components.pm2_5'] = predicted_pm25 if predicted_pm25 is not None else latest_pm25_from_api
        
        # Populate other features...
        if 'components.co' in aqi_used_features: aqi_row_dict['components.co'] = current_components.get('co', 0.0)
        if 'components.no2' in aqi_used_features: aqi_row_dict['components.no2'] = current_components.get('no2', 0.0)
        if 'components.o3' in aqi_used_features: aqi_row_dict['components.o3'] = current_components.get('o3', 0.0)
        if 'coord.lat' in aqi_used_features: aqi_row_dict['coord.lat'] = lat
        if 'coord.lon' in aqi_used_features: aqi_row_dict['coord.lon'] = lon
        if 'aqi_lag1' in aqi_used_features: aqi_row_dict['aqi_lag1'] = current_aqi_lag1
        if 'aqi_lag2' in aqi_used_features: aqi_row_dict['aqi_lag2'] = current_aqi_lag2
        city_feature_name = next((f for f in ['city_name', 'city_encoded'] if f in aqi_used_features), None)
        if city_feature_name: aqi_row_dict[city_feature_name] = encoded_city

        aqi_df = pd.DataFrame([aqi_row_dict])
        aqi_df = create_time_features(aqi_df)
        aqi_df_ordered = aqi_df[aqi_used_features]
        
        model_prediction_raw = aqi_model.predict(aqi_df_ordered)[0]
        predicted_aqi_value = round(float(model_prediction_raw))
        predicted_aqi_value = max(1, min(5, predicted_aqi_value))

        forecast_results_data.append({
            "date": future_date.isoformat(), "predicted_aqi": predicted_aqi_value,
            "predicted_aqi_raw": model_prediction_raw, "predicted_pm25": predicted_pm25,
            "category": classify_aqi_openweathermap(predicted_aqi_value),
        })

        actual_aqi_for_metrics.append(latest_aqi_from_api)
        predicted_aqi_for_metrics.append(predicted_aqi_value)
        if predicted_pm25 is not None:
            actual_pm25_for_metrics.append(latest_pm25_from_api)
            predicted_pm25_for_metrics.append(predicted_pm25)

        current_aqi_lag2 = current_aqi_lag1
        current_aqi_lag1 = float(predicted_aqi_value)
        if predicted_pm25 is not None:
            current_pm25_lag2 = current_pm25_lag1
            current_pm25_lag1 = predicted_pm25

    aqi_mae = mean_absolute_error(actual_aqi_for_metrics, predicted_aqi_for_metrics)
    aqi_rmse = np.sqrt(mean_squared_error(actual_aqi_for_metrics, predicted_aqi_for_metrics))
    
    pm25_mae, pm25_rmse = None, None
    if actual_pm25_for_metrics:
        pm25_mae = mean_absolute_error(actual_pm25_for_metrics, predicted_pm25_for_metrics)
        pm25_rmse = np.sqrt(mean_squared_error(actual_pm25_for_metrics, predicted_pm25_for_metrics))

    return (forecast_results_data, 
            (aqi_mae, aqi_rmse), 
            (pm25_mae, pm25_rmse),
            latest_aqi_from_api, latest_pm25_from_api)

def clear_results():
    output_text.delete(1.0, tk.END)
    for widget in plot_frame.winfo_children(): widget.destroy()
    for widget in plot_pm25_frame.winfo_children(): widget.destroy()
    for widget in plot_performance_frame.winfo_children(): widget.destroy()

def show_forecast():
    city_input = city_combobox.get().strip()
    if not city_input:
        messagebox.showerror("Input Error", "Please select a city.")
        return

    # Clear previous results at the beginning
    clear_results()

    result = forecast_aqi_logic(city_input)
    if isinstance(result, dict) and "error" in result:
        messagebox.showerror("Error", result["error"])
        return

    forecast_data, aqi_metrics, pm25_metrics, latest_aqi_api_val, latest_pm25_api_val = result
    aqi_mae, aqi_rmse = aqi_metrics
    pm25_mae, pm25_rmse = pm25_metrics

    # --- Populate Text Details ---
    if latest_aqi_api_val is not None:
        output_text.insert(tk.END, f"Real-time AQI for {city_input.title()}: {int(latest_aqi_api_val)}\n")
        output_text.insert(tk.END, f"Category: {classify_aqi_openweathermap(latest_aqi_api_val)}\n")
    if latest_pm25_api_val is not None:
        output_text.insert(tk.END, f"Real-time PM2.5: {latest_pm25_api_val:.2f} µg/m³\n\n")

    output_text.insert(tk.END, "Forecast:\n")
    for item in forecast_data:
        output_text.insert(tk.END, f"Date: {item['date']}\n")
        output_text.insert(tk.END, f"Predicted AQI: {item['predicted_aqi']} (raw: {item['predicted_aqi_raw']:.2f})\n")
        if item['predicted_pm25'] is not None:
            output_text.insert(tk.END, f"Predicted PM2.5: {item['predicted_pm25']:.2f} µg/m³\n")
        output_text.insert(tk.END, f"Category: {item['category']}\n\n")

    output_text.insert(tk.END, "AQI Forecast Drift (vs Current):\n")
    output_text.insert(tk.END, f"  MAE: {aqi_mae:.2f}, RMSE: {aqi_rmse:.2f}\n\n")
    
    if pm25_mae is not None:
        output_text.insert(tk.END, "PM2.5 Forecast Drift (vs Current):\n")
        output_text.insert(tk.END, f"  MAE: {pm25_mae:.2f}, RMSE: {pm25_rmse:.2f}\n")

    # --- Visualizations ---
    dates = [datetime.strptime(f['date'], '%Y-%m-%d') for f in forecast_data]

    # --- AQI Forecast Visualization ---
    predicted_aqi_values = [f['predicted_aqi'] for f in forecast_data]
    fig_forecast, ax_forecast = plt.subplots(figsize=(6, 3.5))
    ax_forecast.plot(dates, predicted_aqi_values, marker='o', linestyle='--', color='blue', label='Predicted AQI')
    if latest_aqi_api_val is not None:
        ax_forecast.plot(dates, [latest_aqi_api_val] * len(dates), linestyle='-', color='red', label=f'Real-time AQI: {int(latest_aqi_api_val)}')
    ax_forecast.set_title(f"3-Day AQI Forecast")
    ax_forecast.set_ylabel("AQI (1-5)")
    ax_forecast.grid(True, alpha=0.6)
    ax_forecast.set_yticks(np.arange(1, 6, 1)); ax_forecast.set_ylim(bottom=0.5, top=5.5)
    ax_forecast.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_forecast.legend(loc='best')
    fig_forecast.tight_layout()
    FigureCanvasTkAgg(fig_forecast, master=plot_frame).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- PM2.5 Forecast Visualization ---
    predicted_pm25_values = [f['predicted_pm25'] for f in forecast_data if f['predicted_pm25'] is not None]
    if predicted_pm25_values:
        fig_pm25, ax_pm25 = plt.subplots(figsize=(6, 3.5))
        ax_pm25.plot(dates, predicted_pm25_values, marker='s', linestyle='--', color='green', label='Predicted PM2.5')
        if latest_pm25_api_val is not None:
            ax_pm25.plot(dates, [latest_pm25_api_val] * len(dates), linestyle='-', color='orange', label=f'Real-time PM2.5: {latest_pm25_api_val:.2f}')
        ax_pm25.set_title(f"3-Day PM2.5 Forecast")
        ax_pm25.set_ylabel("PM2.5 (µg/m³)")
        ax_pm25.grid(True, alpha=0.6)
        ax_pm25.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax_pm25.legend(loc='best')
        fig_pm25.tight_layout()
        FigureCanvasTkAgg(fig_pm25, master=plot_pm25_frame).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Performance Visualization ---
    fig_perf, (ax_aqi_perf, ax_pm25_perf) = plt.subplots(1, 2, figsize=(8, 3), sharey=False)
    
    # AQI Drift Plot
    aqi_metrics_names, aqi_metrics_values = ['MAE', 'RMSE'], [aqi_mae, aqi_rmse]
    ax_aqi_perf.bar(aqi_metrics_names, aqi_metrics_values, color=['cornflowerblue', 'salmon'])
    ax_aqi_perf.set_title("AQI Drift")
    ax_aqi_perf.set_ylabel("Value")
    for i, v in enumerate(aqi_metrics_values): ax_aqi_perf.text(i, v, f'{v:.2f}', ha='center', va='bottom')

    # PM2.5 Drift Plot
    if pm25_mae is not None:
        pm25_metrics_names, pm25_metrics_values = ['MAE', 'RMSE'], [pm25_mae, pm25_rmse]
        ax_pm25_perf.bar(pm25_metrics_names, pm25_metrics_values, color=['cornflowerblue', 'salmon'])
        ax_pm25_perf.set_title("PM2.5 Drift")
        for i, v in enumerate(pm25_metrics_values): ax_pm25_perf.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    else:
        ax_pm25_perf.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax_pm25_perf.set_title("PM2.5 Drift")
        ax_pm25_perf.set_xticks([])
        ax_pm25_perf.set_yticks([])

    fig_perf.tight_layout(pad=2.0)
    FigureCanvasTkAgg(fig_perf, master=plot_performance_frame).get_tk_widget().pack(fill=tk.BOTH, expand=True)


# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("AERIS - AQI & PM2.5 Forecast")
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)
control_frame = ttk.Frame(main_frame)
control_frame.pack(pady=5, fill=tk.X)
city_label = tk.Label(control_frame, text="Select City:")
city_label.pack(side=tk.LEFT, padx=5)
city_combobox = ttk.Combobox(control_frame, values=cities_list, width=30)
if cities_list: city_combobox.current(0)
city_combobox.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
forecast_button = tk.Button(control_frame, text="Get Forecast", command=show_forecast)
forecast_button.pack(side=tk.LEFT, padx=5)
clear_button = tk.Button(control_frame, text="Clear", command=clear_results)
clear_button.pack(side=tk.LEFT, padx=5)

results_frame = ttk.Frame(main_frame)
results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
text_frame = ttk.LabelFrame(results_frame, text="Forecast Details", padding="10")
text_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, anchor='n')
output_text = tk.Text(text_frame, height=25, width=55)
output_text.pack(fill=tk.Y, expand=True)

plot_container_frame = ttk.Frame(results_frame)
plot_container_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
plot_frame = ttk.LabelFrame(plot_container_frame, text="AQI Forecast")
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
plot_pm25_frame = ttk.LabelFrame(plot_container_frame, text="PM2.5 Forecast")
plot_pm25_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
plot_performance_frame = ttk.LabelFrame(plot_container_frame, text="Forecast Drift from Current Value")
plot_performance_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

root.mainloop()