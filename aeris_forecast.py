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
import matplotlib.dates as mdates # Added for date formatting on plot


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
    # EPA PM2.5 breakpoints (24-hour)
    if value <= 12.0: return "Good"
    elif value <= 35.4: return "Moderate"
    elif value <= 55.4: return "Poor" # Originally "Unhealthy for Sensitive Groups"
    # elif value <= 150.4: return "Unhealthy" # Added for more granularity if needed
    # elif value <= 250.4: return "Very Unhealthy" # Added
    else: return "Hazardous"

def create_time_features(df):
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
    # df['hour'] = df['datetime'].dt.hour # 'hour' is not in the new used_features
    df['day_of_week'] = df['datetime'].dt.dayofweek # Matches 'day_of_week' in used_features
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    return df

# Ensure you have your actual API key here
OPENWEATHERMAP_API_KEY = "APIKEY" # Replace with your actual API key
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

    # Check if data['list'] exists and is not empty
    if 'list' not in data or not data['list']:
         raise ValueError(f"Unexpected API response format for {city_name_input}: 'list' is missing or empty.")

    # Check if 'components' exists in the first item of the list
    if 'components' not in data['list'][0]:
        raise ValueError(f"Unexpected API response format for {city_name_input}: 'components' missing in data list item.")


    # Return the entire 'components' dictionary
    return data['list'][0]['components']

def forecast_pm2_5(city):
    forecast_results = []
    actual_values_for_metrics = [] # This will contain the initial real-time value repeated for each forecast day
    predicted_values_for_metrics = [] # This will contain the predicted values for each forecast day


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
        # Find the exact case-sensitive city name from the encoder classes
        correctly_cased_city = next(c for c in city_encoder.classes_ if c.lower() == city_lower)
        encoded_city = city_encoder.transform([correctly_cased_city])[0]
    except StopIteration:
        return {"error": f"City '{city}' not recognized by city encoder. Please check the city name."}
    except Exception as e:
         return {"error": f"Error encoding city '{city}': {e}"}


    current_processing_date = datetime.now().date()

    for i in range(1, 4): # Forecast for next 3 days (Day 1, Day 2, Day 3)
        future_date = current_processing_date + timedelta(days=i)
        # Assuming prediction is for the start of the day or average of the day
        forecast_datetime_obj = datetime.combine(future_date, datetime.min.time())


        row_dict = {col: 0.0 for col in used_features} # Initialize with defaults

        # 1. Populate component features (using most recent API fetch for all forecast days)
        #    Assumption: these components are relatively stable or their future values are not being predicted.
        for comp_feature in ['components.pm10', 'components.no2', 'components.so2', 'components.co', 'components.o3', 'components.no', 'components.nh3']:
            if comp_feature in used_features:
                # Extract the component name (e.g., 'pm10' from 'components.pm10')
                comp_name = comp_feature.split('.')[-1]
                # Use .get() with a default value to avoid KeyError if a component is missing
                row_dict[comp_feature] = current_air_components.get(comp_name, 0.0)


        # 2. Populate time-related features (will be done by create_time_features later)
        row_dict['datetime'] = forecast_datetime_obj # For create_time_features

        # 3. Populate lag features
        if 'pm2_5_lag1' in used_features:
            row_dict['pm2_5_lag1'] = current_pm2_5_lag1
        if 'pm2_5_lag2' in used_features:
            row_dict['pm2_5_lag2'] = current_pm2_5_lag2

        # 4. Populate city feature
        #    Assumption: Model was trained with encoded city values under the column name 'city_name'
        #    Adjust if your model expects a different column name for the encoded city.
        city_feature_name = None
        if 'city_name' in used_features:
            city_feature_name = 'city_name'
        elif 'city_encoded' in used_features: # Fallback if an older name was used
             city_feature_name = 'city_encoded'

        if city_feature_name:
            row_dict[city_feature_name] = encoded_city
        else:
             # Handle case where neither expected city feature name is in used_features
             # This might indicate a mismatch between the model/features file and the expected features
             return {"error": f"City feature ('city_name' or 'city_encoded') not found in used_features."}


        # Create DataFrame for this single forecast point
        row_df = pd.DataFrame([row_dict])
        row_df = create_time_features(row_df) # Adds 'month', 'day_of_week', 'is_weekend'

        # Ensure all expected features are present in row_df before selection
        # This is crucial: `used_features` names must match columns available in `row_df`
        missing_in_row_df = [col for col in used_features if col not in row_df.columns]
        if missing_in_row_df:
            # print(f"Debug: Missing features in row_df: {missing_in_row_df}")
            # print(f"Debug: row_df columns: {row_df.columns.tolist()}")
            return {"error": f"Feature engineering mismatch. Missing required features: {missing_in_row_df}"}

        try:
            # Select and order features according to what the model expects
            row_df_ordered = row_df[used_features]
        except KeyError as e:
            return {"error": f"KeyError during feature selection: {e}. Check `used_features` and columns created for forecasting."}

        # Double-check feature count just before prediction
        if row_df_ordered.shape[1] != model.num_feature():
             return {"error": f"Feature count mismatch! Prepared data has {row_df_ordered.shape[1]} features, but the model expects {model.num_feature()}."}


        # Make prediction
        model_prediction_raw = model.predict(row_df_ordered)[0]
        # Ensure prediction is non-negative
        model_prediction_processed = max(0.0, float(model_prediction_raw))

        # Blending/Correction (optional, you might want to adjust or remove this)
        # Using latest_pm2_5_value from the initial API call for blending across all days.
        # A more sophisticated approach might use a decay factor or a time-series model output.
        # Example simple blending: weighted average of current value and model prediction
        # This simple blending might not be ideal for forecasting multiple days ahead
        # corrected_prediction = 0.8 * latest_pm2_5_value + 0.2 * model_prediction_processed # Example blending
        # For simplicity here, we'll just use the processed model prediction
        predicted_pm2_5 = round(model_prediction_processed, 2)


        forecast_results.append({
            "date": future_date.isoformat(),
            "predicted_pm2_5": predicted_pm2_5,
            "category": classify_pm2_5(predicted_pm2_5),
        })

        # For performance metrics, we compare each day's prediction against the *initial* real-time value.
        # This gives an idea of how the forecast degrades over days compared to the starting point.
        actual_values_for_metrics.append(latest_pm2_5_value)
        predicted_values_for_metrics.append(predicted_pm2_5)


        # Update lags for the next iteration:
        # The PM2.5 predicted for today becomes lag1 for tomorrow.
        # Today's lag1 becomes lag2 for tomorrow.
        # Using the current prediction as the basis for the next day's lag
        current_pm2_5_lag2 = current_pm2_5_lag1
        current_pm2_5_lag1 = predicted_pm2_5


    if not actual_values_for_metrics or not predicted_values_for_metrics or len(actual_values_for_metrics) != len(predicted_values_for_metrics):
        return {"error": "Error generating forecast or inconsistent results for metrics calculation."}

    # Calculate metrics based on the comparison of predicted values against the initial real-time value
    # over the 3 forecast days.
    try:
        mae = mean_absolute_error(actual_values_for_metrics, predicted_values_for_metrics)
        rmse = np.sqrt(mean_squared_error(actual_values_for_metrics, predicted_values_for_metrics))

        # --- START of R2 Calculation Fix ---
        # Calculate R² only if there is variance in the actual values for metrics
        # (i.e., if actual_values_for_metrics is not just a list of the same value)
        # This is important because R² is undefined when the true values have zero variance.
        if np.var(actual_values_for_metrics) > 1e-9: # Use a small tolerance for floating point comparison
             r2 = r2_score(actual_values_for_metrics, predicted_values_for_metrics)
        else:
             r2 = None # Set R2 to None or a specific indicator if not applicable
        # --- END of R2 Calculation Fix ---

    except Exception as e:
        # Catch any other potential errors during metrics calculation
        return {"error": f"Error calculating metrics: {e}"}


    return forecast_results, mae, rmse, r2, latest_pm2_5_value

# --- START of Clear Results Function ---
def clear_results():
    # Clear the text output
    output_text.delete(1.0, tk.END)

    # Clear the forecast plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Clear the performance plot
    for widget in plot_performance_frame.winfo_children():
        widget.destroy()
# --- END of Clear Results Function ---


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

    output_text.insert(tk.END, f"Model Performance (vs initial real-time PM2.5 over 3 days):\n")
    output_text.insert(tk.END, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\n")
    # Display R2 or N/A based on the calculation result
    if r2 is not None:
        output_text.insert(tk.END, f"R²: {r2:.2f}\n")
    else:
        output_text.insert(tk.END, "R²: N/A (Not applicable for constant 'actual' baseline)\n")


    # --- START of Forecast Visualization Code ---

    # Plotting PM2.5 forecast
    dates = [datetime.strptime(f['date'], '%Y-%m-%d') for f in forecast_results]
    predicted_values = [f['predicted_pm2_5'] for f in forecast_results]
    # categories = [f['category'] for f in forecast_results] # Not directly used for point colors in this version

    # Clear previous forecast plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    fig_forecast, ax_forecast = plt.subplots(figsize=(7, 4)) # Adjusted size slightly for legend

    # Plot the Predicted PM2.5 line with dashed style and markers
    ax_forecast.plot(dates, predicted_values, marker='o', linestyle='--', color='blue', label='Predicted PM2.5')

    # Plot the Real-time PM2.5 as a horizontal reference line across the forecast dates
    # Create a list of the latest_pm2_5_api value repeated for each date to plot a horizontal line
    realtime_values = [latest_pm2_5_api] * len(dates)
    ax_forecast.plot(dates, realtime_values, linestyle='-', color='red', label='Real-time PM2.5 (Reference)')


    ax_forecast.set_title(f"3-Day PM2.5 Forecast for {city_input.title()}")
    ax_forecast.set_xlabel("Date")
    ax_forecast.set_ylabel("PM2.5 (µg/m³)")
    ax_forecast.grid(True, linestyle='--', alpha=0.6) # Add grid lines
    ax_forecast.set_ylim(bottom=0) # Ensure y-axis starts at 0


    # --- Start of Improved X-axis Formatting ---

    # Set the x-axis tick locations to be exactly the forecast dates
    ax_forecast.set_xticks(dates)

    # Define a simple date format (e.g., 'Month Day, Year')
    date_form = mdates.DateFormatter('%B %d, %Y')
    ax_forecast.xaxis.set_major_formatter(date_form)

    # Improve rotation and alignment of dates for readability
    fig_forecast.autofmt_xdate(rotation=45, ha='right')
    # --- End of Improved X-axis Formatting ---


    # Add value labels on the Predicted PM2.5 line
    for i, value in enumerate(predicted_values):
        # Adjust text position slightly if values are close to the reference line
        vertical_offset = 1
        if abs(value - latest_pm2_5_api) < 2: # Adjust offset if values are very close
             vertical_offset = -1 if value > latest_pm2_5_api else 1
        ax_forecast.text(dates[i], value + vertical_offset, f"{value:.2f}", ha='center', va='bottom' if vertical_offset > 0 else 'top', fontsize=8, color='blue')

    # Add a label for the Real-time reference value
    # Place this label at the start of the line
    # Check if dates list is not empty before trying to access dates[0]
    if dates:
        ax_forecast.text(dates[0], latest_pm2_5_api, f"{latest_pm2_5_api:.2f}", ha='right', va='center', fontsize=8, color='red')


    # Add a legend to explain the lines
    # Use bbox_to_anchor and loc to place the legend outside the plot area if it overlaps
    ax_forecast.legend(loc='upper left', bbox_to_anchor=(1, 1))


    fig_forecast.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent legend overlapping plot
    canvas_forecast = FigureCanvasTkAgg(fig_forecast, master=plot_frame)
    canvas_forecast.draw()
    canvas_forecast.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- END of Forecast Visualization Code ---


    # --- START of Performance Visualization Code ---

    # Plotting Model Performance Metrics
    metrics_names = ['MAE', 'RMSE']
    metrics_values = [mae, rmse]
    # Add R2 to metrics if it was calculated
    if r2 is not None:
        metrics_names.append('R²')
        metrics_values.append(r2)


    # Clear previous performance plot
    for widget in plot_performance_frame.winfo_children():
        widget.destroy()

    fig_perf, ax_perf = plt.subplots(figsize=(7, 4)) # Adjusted size to match forecast plot
    # Using different colors for clarity
    colors = ['salmon', 'lightgreen', 'cornflowerblue']
    # Ensure colors match the metrics_names list length
    colors = colors[:len(metrics_names)]

    bars = ax_perf.bar(metrics_names, metrics_values, color=colors)
    ax_perf.set_title("Model Performance Evaluation (vs initial real-time)")
    ax_perf.set_ylabel("Value")
    # Adjust y-axis limits to handle negative R2 if it's present, otherwise ensure it starts at 0
    min_val = min(metrics_values) if metrics_values else 0
    max_val = max(metrics_values) if metrics_values else 0

    # Add some padding to the y-axis limits
    # If only MAE/RMSE (>=0), start y_bottom at 0. If R2 is included and negative, adjust y_bottom.
    y_bottom = min(0, min_val - abs(min_val)*0.2) if min_val <= 0 and 'R²' in metrics_names else 0

    y_top = max_val + abs(max_val)*0.2 if max_val >= 0 else 0.1 # Add a small upper limit if max_val is 0

    # Ensure top is greater than bottom, add a small buffer if they are equal or close
    if y_top <= y_bottom:
         y_top = y_bottom + 0.1 # Ensure a minimum range


    ax_perf.set_ylim(bottom=y_bottom, top=y_top)


    # Add value labels on the performance plot
    for bar in bars:
        yval = bar.get_height()
        # Adjust text position based on value sign
        va = 'bottom' if yval >= 0 else 'top'
        # Calculate a dynamic offset based on the current bar height and overall axis range
        # Avoid division by zero if range is zero (shouldn't happen with adjusted limits but safe)
        y_range = ax_perf.get_ylim()[1] - ax_perf.get_ylim()[0]
        offset = 0.02 * y_range if y_range > 0 else 0.1 # 2% of the y-axis range as offset
        ax_perf.text(bar.get_x() + bar.get_width()/2.0, yval + (offset if yval >= 0 else -offset), f'{yval:.2f}', ha='center', va=va, fontsize=8)


    fig_perf.tight_layout()
    canvas_perf = FigureCanvasTkAgg(fig_perf, master=plot_performance_frame)
    canvas_perf.draw()
    canvas_perf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- END of Performance Visualization Code ---


# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("AERIS - Real-time Air Quality Forecasting")

# Create a main frame to hold everything
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Create a frame for controls (city selection, forecast button, clear button)
control_frame = ttk.Frame(main_frame)
control_frame.pack(pady=5)

city_label = tk.Label(control_frame, text="Select City:")
city_label.pack(side=tk.LEFT, padx=5)

city_combobox = ttk.Combobox(control_frame, values=cities_list, width=30)
city_combobox.pack(side=tk.LEFT, padx=5)

forecast_button = tk.Button(control_frame, text="Get Forecast", command=show_forecast)
forecast_button.pack(side=tk.LEFT, padx=5)

# --- START of Clear Button ---
clear_button = tk.Button(control_frame, text="Clear", command=clear_results)
clear_button.pack(side=tk.LEFT, padx=5)
# --- END of Clear Button ---

# Create a frame for text output and plot container
results_frame = ttk.Frame(main_frame)
results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

# Frame for text output
text_frame = ttk.LabelFrame(results_frame, text="Forecast Details", padding="10")
text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

output_text = tk.Text(text_frame, height=15, width=50) # Increased height slightly
output_text.pack(fill=tk.BOTH, expand=True)

# Frame to hold both plots side by side
plot_container_frame = ttk.Frame(results_frame) # New container frame
plot_container_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5) # Pack to the left of text_frame

# Frame for forecast plot (modified to be inside plot_container_frame)
plot_frame = tk.Frame(plot_container_frame)
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Pack at the top within the container

# Frame for performance plot (new)
plot_performance_frame = tk.Frame(plot_container_frame)
plot_performance_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True) # Pack at the bottom within the container


root.mainloop()