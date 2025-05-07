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
    with open('pm2_5_forecasting_model.pkl', 'rb') as f:
        model: Booster = pickle.load(f)
    # IMPORTANT: Ensure this is the correct pkl file with 13 features
    with open('used_features.pkl', 'rb') as f: # Assuming you renamed it or this is the correct one
        used_features = pickle.load(f)
    city_encoder = joblib.load("city_label_encoder.pkl")
    city_coords = pd.read_csv('city_coordinates_from_dataset.csv')

    cities_list = city_coords['city_name'].tolist()

    if model:
        pass # Model loaded successfully.

except FileNotFoundError as e:
    print(f"Error: A required file was not found: {e.filename}")
    print("Please ensure required files ('pm2_5_forecasting_model.pkl', 'used_features.pkl', 'city_label_encoder.pkl', 'city_coordinates_from_dataset.csv') are in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An error occurred during initial file loading: {e}")
    exit()

def classify_pm2_5(value):
    if value is None: return "Unknown"
    if value <= 12: return "Good"
    elif value <= 35.4: return "Moderate"
    elif value <= 55.4: return "Poor"
    else: return "Hazardous"

def create_time_features(df):
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
    # df['hour'] = df['datetime'].dt.hour # 'hour' is not in the new used_features
    df['day_of_week'] = df['datetime'].dt.dayofweek # Matches 'day_of_week' in used_features
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    return df

OPENWEATHERMAP_API_KEY = "046db00705c9a85bd3daa1f61ea04d4a" # Replace with your actual key
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

    if 'list' not in data or not data['list'] or 'components' not in data['list'][0]:
        raise ValueError(f"Unexpected API response format for {city_name_input}.")

    # Return the entire 'components' dictionary
    return data['list'][0]['components']

def forecast_pm2_5(city):
    forecast_results = []
    actual_values_for_metrics = []
    predicted_values_for_metrics = []

    try:
        current_air_components = fetch_realtime_air_data(city)
        latest_pm2_5_value = current_air_components.get('pm2_5')

        if latest_pm2_5_value is None:
            return {"error": f"PM2.5 data not found in API response for {city.title()}."}

        # Initialize lags for recursive forecasting
        # For the first day, pm2_5_lag1 is the current PM2.5, pm2_5_lag2 can also be set to current PM2.5
        # or a historical value if available. We'll use current for simplicity here.
        current_pm2_5_lag1 = latest_pm2_5_value
        current_pm2_5_lag2 = latest_pm2_5_value # Assumption for the very first lag2 value

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

    current_processing_date = datetime.now().date()

    for i in range(1, 4): # Forecast for next 3 days
        future_date = current_processing_date + timedelta(days=i)
        forecast_datetime_obj = datetime.combine(future_date, datetime.min.time())

        row_dict = {col: 0.0 for col in used_features} # Initialize with defaults

        # 1. Populate component features (using most recent API fetch for all forecast days)
        #    Assumption: these components are relatively stable or their future values are not being predicted.
        for comp_feature in ['components.pm10', 'components.no2', 'components.so2', 'components.co', 'components.o3', 'components.no', 'components.nh3']:
            if comp_feature in used_features:
                row_dict[comp_feature] = current_air_components.get(comp_feature.split('.')[-1], 0.0) # e.g. components.pm10 -> pm10

        # 2. Populate time-related features (will be done by create_time_features later)
        row_dict['datetime'] = forecast_datetime_obj # For create_time_features

        # 3. Populate lag features
        if 'pm2_5_lag1' in used_features:
            row_dict['pm2_5_lag1'] = current_pm2_5_lag1
        if 'pm2_5_lag2' in used_features:
            row_dict['pm2_5_lag2'] = current_pm2_5_lag2

        # 4. Populate city feature
        #    Assumption: Model was trained with encoded city values under the column name 'city_name'
        if 'city_name' in used_features:
            row_dict['city_name'] = encoded_city
        elif 'city_encoded' in used_features: # Fallback if old name still there
            row_dict['city_encoded'] = encoded_city


        row_df = pd.DataFrame([row_dict])
        row_df = create_time_features(row_df) # Adds 'month', 'day_of_week', 'is_weekend'

        # Ensure all expected features are present in row_df before selection
        # This is crucial: `used_features` names must match columns available in `row_df`
        missing_in_row_df = [col for col in used_features if col not in row_df.columns]
        if missing_in_row_df:
            return {"error": f"Feature engineering mismatch. Missing: {missing_in_row_df}"}

        try:
            row_df_ordered = row_df[used_features]
        except KeyError as e:
            return {"error": f"KeyError during feature selection: {e}. Check `used_features` and `row_df` columns."}

        if row_df_ordered.shape[1] != model.num_feature():
            return {"error": f"Feature count mismatch! Data has {row_df_ordered.shape[1]}, model expects {model.num_feature()}."}

        model_prediction_raw = model.predict(row_df_ordered)[0]
        model_prediction_processed = max(0.0, float(model_prediction_raw))

        # Blending/Correction (optional, you might want to adjust or remove this)
        # Using latest_pm2_5_value from the initial API call for blending across all days.
        corrected_prediction = 0.8 * latest_pm2_5_value + 0.2 * model_prediction_processed
        predicted_pm2_5 = round(max(0.0, corrected_prediction), 2)

        forecast_results.append({
            "date": future_date.isoformat(),
            "predicted_pm2_5": predicted_pm2_5,
            "category": classify_pm2_5(predicted_pm2_5),
        })

        actual_values_for_metrics.append(latest_pm2_5_value) # Using initial real-time as reference
        predicted_values_for_metrics.append(predicted_pm2_5)

        # Update lags for the next iteration:
        # The PM2.5 predicted for today becomes lag1 for tomorrow.
        # Today's lag1 becomes lag2 for tomorrow.
        current_pm2_5_lag2 = current_pm2_5_lag1
        current_pm2_5_lag1 = predicted_pm2_5 # Use the blended/corrected prediction as the basis for next lag

    if not actual_values_for_metrics or not predicted_values_for_metrics:
        return {"error": "No forecast values generated."}

    mae = mean_absolute_error(actual_values_for_metrics, predicted_values_for_metrics)
    rmse = np.sqrt(mean_squared_error(actual_values_for_metrics, predicted_values_for_metrics))
    r2 = r2_score(actual_values_for_metrics, predicted_values_for_metrics)

    return forecast_results, mae, rmse, r2, latest_pm2_5_value

def show_forecast():
    city_input = city_combobox.get().strip()
    if not city_input:
        messagebox.showerror("Input Error", "Please select a city.")
        return

    result = forecast_pm2_5(city_input)
    if isinstance(result, dict) and "error" in result:
        messagebox.showerror("Error", result["error"])
        return

    forecast_results, mae, rmse, r2, latest_pm2_5_api = result

    # Display text output
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"Real-time PM2.5 for {city_input.title()}: {latest_pm2_5_api:.2f} µg/m³\n\n")
    for forecast in forecast_results:
        output_text.insert(tk.END, f"Date: {forecast['date']}\n")
        output_text.insert(tk.END, f"Predicted PM2.5: {forecast['predicted_pm2_5']} µg/m³\n")
        output_text.insert(tk.END, f"Category: {forecast['category']}\n\n")

    output_text.insert(tk.END, f"Model Performance (vs initial real-time PM2.5):\n")
    output_text.insert(tk.END, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}\n")

    # Generate and display plot
    dates = [datetime.strptime(f['date'], '%Y-%m-%d') for f in forecast_results]
    predicted_values = [f['predicted_pm2_5'] for f in forecast_results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dates, predicted_values, marker='o', linestyle='-', color='blue')
    ax.set_title(f"3-Day PM2.5 Forecast for {city_input.title()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.grid(True)
    fig.autofmt_xdate() # Auto-format dates on x-axis

    # Embed plot in Tkinter window
    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


root = tk.Tk()
root.title("AERIS - Real-time Air Quality Forecasting")

# Create a main frame to hold everything
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Create a frame for controls (city selection and button)
control_frame = ttk.Frame(main_frame)
control_frame.pack(pady=5)

city_label = tk.Label(control_frame, text="Select City:")
city_label.pack(side=tk.LEFT, padx=5)

city_combobox = ttk.Combobox(control_frame, values=cities_list, width=30)
city_combobox.pack(side=tk.LEFT, padx=5)

forecast_button = tk.Button(control_frame, text="Get Forecast", command=show_forecast)
forecast_button.pack(side=tk.LEFT, padx=5)

# Create a frame for text output and plot
results_frame = ttk.Frame(main_frame)
results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

# Frame for text output
text_frame = ttk.LabelFrame(results_frame, text="Forecast Details", padding="10")
text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

output_text = tk.Text(text_frame, height=12, width=60)
output_text.pack(fill=tk.BOTH, expand=True)

# Frame for plot
plot_frame = ttk.LabelFrame(results_frame, text="Forecast Visualization", padding="10")
plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

root.mainloop()