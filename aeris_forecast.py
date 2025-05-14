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
    # Ensure value is an integer for classification after prediction
    val_int = round(float(value))
    if val_int == 1: return "Good"
    elif val_int == 2: return "Fair"
    elif val_int == 3: return "Moderate"
    elif val_int == 4: return "Poor"
    elif val_int >= 5: return "Very Poor" # Treat 5 and above as "Very Poor"
    else: return "Unknown" # Should ideally not happen if input is 1-5

def create_time_features(df):
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    return df

OPENWEATHERMAP_API_KEY = "APIKEY" # Replace actual API key here
# Ensure to set your OpenWeatherMap API key here
# OpenWeatherMap API URL for air pollution data
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
    if 'list' not in data or not data['list'] or 'main' not in data['list'][0] or 'aqi' not in data['list'][0]['main']:
        raise ValueError(f"Unexpected API response format for {city_name_input}: 'main.aqi' missing.")
    return data['list'][0]['main']['aqi'] # This is an integer from 1 to 5

def fetch_weather_forecast(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    return response.json()

def extract_daily_weather_averages(forecast_data):
    daily_data = defaultdict(lambda: {'temp': [], 'humidity': [], 'wind': []})
    for entry in forecast_data['list']:
        dt = datetime.fromtimestamp(entry['dt'])
        day_str = dt.date().isoformat()
        daily_data[day_str]['temp'].append(entry['main']['temp'])
        daily_data[day_str]['humidity'].append(entry['main']['humidity'])
        daily_data[day_str]['wind'].append(entry['wind']['speed'])
    daily_averages = {}
    for day, vals in daily_data.items():
        daily_averages[day] = {
            'avg_temp': sum(vals['temp']) / len(vals['temp']),
            'avg_humidity': sum(vals['humidity']) / len(vals['humidity']),
            'avg_wind': sum(vals['wind']) / len(vals['wind']),
        }
    return daily_averages

def forecast_aqi_logic(city):
    forecast_results_data = []
    actual_values_for_metrics = []
    predicted_values_for_metrics = []
    latest_aqi_from_api = None

    try:
        current_aqi_value = fetch_realtime_air_data(city) # Expected to be 1-5
        latest_aqi_from_api = float(current_aqi_value)

        current_aqi_lag1 = latest_aqi_from_api
        current_aqi_lag2 = latest_aqi_from_api
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error during data fetching: {str(e)}"}

    city_lower = city.lower()
    try:
        correctly_cased_city = next(c for c in city_encoder.classes_ if c.lower() == city_lower)
        encoded_city = city_encoder.transform([correctly_cased_city])[0]
    except StopIteration:
        return {"error": f"City '{city}' not recognized by city encoder."}
    except Exception as e:
         return {"error": f"Error encoding city '{city}': {e}"}

    current_processing_date = datetime.now().date()

    for i in range(1, 4):
        future_date = current_processing_date + timedelta(days=i)
        forecast_datetime_obj = datetime.combine(future_date, datetime.min.time())
        row_dict = {col: 0.0 for col in used_features}
        row_dict['datetime'] = forecast_datetime_obj

        if 'aqi_lag1' in used_features: row_dict['aqi_lag1'] = current_aqi_lag1
        if 'aqi_lag2' in used_features: row_dict['aqi_lag2'] = current_aqi_lag2
        
        city_feature_name = None
        if 'city_name' in used_features: city_feature_name = 'city_name'
        elif 'city_encoded' in used_features: city_feature_name = 'city_encoded'
        if city_feature_name: row_dict[city_feature_name] = encoded_city
        else: return {"error": "City feature ('city_name' or 'city_encoded') not in used_features."}

        row_df = pd.DataFrame([row_dict])
        row_df = create_time_features(row_df)
        missing_in_row_df = [col for col in used_features if col not in row_df.columns]
        if missing_in_row_df:
            return {"error": f"Missing required features for model: {missing_in_row_df}"}
        try:
            row_df_ordered = row_df[used_features]
        except KeyError as e:
            return {"error": f"KeyError during feature selection: {e}."}
        if row_df_ordered.shape[1] != model.num_feature():
             return {"error": f"Feature count mismatch! Model expects {model.num_feature()}, got {row_df_ordered.shape[1]}."}

        model_prediction_raw = model.predict(row_df_ordered)[0]
        # Process prediction: round to nearest int, clamp between 1 and 5
        predicted_aqi_value = round(float(model_prediction_raw))
        ##predicted_aqi_value = max(1, min(5, predicted_aqi_value)) # Clamp to 1-5 range

        forecast_results_data.append({
            "date": future_date.isoformat(),
            "predicted_aqi": predicted_aqi_value,
            "category": classify_aqi_openweathermap(predicted_aqi_value), # Use new classification
        })

        actual_values_for_metrics.append(latest_aqi_from_api) # Use the initial API AQI for all days for this metric
        predicted_values_for_metrics.append(predicted_aqi_value)

        current_aqi_lag2 = current_aqi_lag1
        current_aqi_lag1 = float(predicted_aqi_value) # Use the processed predicted value for next lag

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

def clear_results():
    output_text.delete(1.0, tk.END)
    for widget in plot_frame.winfo_children(): widget.destroy()
    for widget in plot_performance_frame.winfo_children(): widget.destroy()

def show_forecast():
    city_input = city_combobox.get().strip()
    if not city_input:
        messagebox.showerror("Input Error", "Please select a city.")
        return

    result = forecast_aqi_logic(city_input)
    if isinstance(result, dict) and "error" in result:
        messagebox.showerror("Error", result["error"])
        return

    forecast_data, mae, rmse, r2, latest_aqi_api_val = result # Renamed for clarity

    output_text.delete(1.0, tk.END)
    if latest_aqi_api_val is not None:
        output_text.insert(tk.END, f"Real-time AQI for {city_input.title()}: {int(latest_aqi_api_val)}\n")
        output_text.insert(tk.END, f"Category: {classify_aqi_openweathermap(latest_aqi_api_val)}\n\n") # Use new classification
    else:
        output_text.insert(tk.END, f"Real-time AQI for {city_input.title()}: Data unavailable\n\n")

    output_text.insert(tk.END, "Forecast:\n")
    for item in forecast_data:
        output_text.insert(tk.END, f"Date: {item['date']}\n")
        output_text.insert(tk.END, f"Predicted AQI: {item['predicted_aqi']}\n") # This is 1-5
        output_text.insert(tk.END, f"Category: {item['category']}\n\n") # Based on 1-5

    output_text.insert(tk.END, "Model Performance (vs initial real-time AQI over 3 days):\n")
    if mae is not None: output_text.insert(tk.END, f"MAE: {mae:.2f}\n")
    if rmse is not None: output_text.insert(tk.END, f"RMSE: {rmse:.2f}\n")
    if r2 is not None:
        if np.isnan(r2): output_text.insert(tk.END, "R²: N/A\n")
        else: output_text.insert(tk.END, f"R²: {r2:.2f}\n")
    else: output_text.insert(tk.END, "Metrics calculation failed.\n")

    # --- Forecast Visualization ---
    dates = [datetime.strptime(f['date'], '%Y-%m-%d') for f in forecast_data]
    predicted_values = [f['predicted_aqi'] for f in forecast_data]

    for widget in plot_frame.winfo_children(): widget.destroy()
    fig_forecast, ax_forecast = plt.subplots(figsize=(7, 4))
    ax_forecast.plot(dates, predicted_values, marker='o', linestyle='--', color='blue', label='Predicted AQI (1-5 scale)')

    if latest_aqi_api_val is not None:
        realtime_plot_vals = [latest_aqi_api_val] * len(dates)
        ax_forecast.plot(dates, realtime_plot_vals, linestyle='-', color='red', label=f'Real-time AQI: {int(latest_aqi_api_val)}')

    ax_forecast.set_title(f"3-Day AQI Forecast for {city_input.title()} (OpenWeatherMap Scale 1-5)")
    ax_forecast.set_xlabel("Date")
    ax_forecast.set_ylabel("AQI (1=Good, 5=Very Poor)")
    ax_forecast.grid(True, linestyle='--', alpha=0.6)
    # Adjust Y-axis for 1-5 scale
    ax_forecast.set_yticks(np.arange(1, 6, 1)) # Ticks at 1, 2, 3, 4, 5
    ax_forecast.set_ylim(bottom=0.5, top=5.5) # Give some padding

    ax_forecast.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig_forecast.autofmt_xdate(rotation=30, ha='right')

    for i, value in enumerate(predicted_values):
        ax_forecast.text(dates[i], value + 0.1, f"{value}", ha='center', va='bottom', fontsize=8, color='blue')
    
    ax_forecast.legend(loc='best')
    fig_forecast.tight_layout()
    canvas_forecast = FigureCanvasTkAgg(fig_forecast, master=plot_frame)
    canvas_forecast.draw()
    canvas_forecast.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Performance Visualization ---
    metrics_names, metrics_values = [], []
    if mae is not None: metrics_names.append('MAE'); metrics_values.append(mae)
    if rmse is not None: metrics_names.append('RMSE'); metrics_values.append(rmse)
    if r2 is not None and not np.isnan(r2): metrics_names.append('R²'); metrics_values.append(r2)
    
    for widget in plot_performance_frame.winfo_children(): widget.destroy()
    if metrics_names:
        fig_perf, ax_perf = plt.subplots(figsize=(7, 4))
        colors = ['salmon', 'lightgreen', 'cornflowerblue'][:len(metrics_names)]
        bars = ax_perf.bar(metrics_names, metrics_values, color=colors)
        ax_perf.set_title("Model Performance (AQI 1-5 scale)")
        ax_perf.set_ylabel("Value")
        min_val = min(metrics_values + [0]) # include 0 for y_bottom if all positive
        max_val = max(metrics_values + [0.1]) # include 0.1 for y_top if all zero or negative
        
        y_bottom_padding = abs(min_val) * 0.1 if min_val < 0 else 0
        y_top_padding = abs(max_val) * 0.1
        
        ax_perf.set_ylim(bottom=min_val - y_bottom_padding, top=max_val + y_top_padding + 0.1) # Ensure some space above bars

        for bar in bars:
            yval = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom' if yval >=0 else 'top', fontsize=8)
        fig_perf.tight_layout()
        canvas_perf = FigureCanvasTkAgg(fig_perf, master=plot_performance_frame)
        canvas_perf.draw()
        canvas_perf.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    else:
        tk.Label(plot_performance_frame, text="Performance metrics not available.").pack(padx=10, pady=10)

# --- Tkinter GUI Setup ---
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

forecast_button = tk.Button(control_frame, text="Get AQI Forecast (1-5)", command=show_forecast)
forecast_button.pack(side=tk.LEFT, padx=5)
clear_button = tk.Button(control_frame, text="Clear", command=clear_results)
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