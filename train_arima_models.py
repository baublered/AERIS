import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

# Load dataset
data = pd.read_csv("final_dataset.csv", parse_dates=["datetime"])

# Fix column names
data = data[["datetime", "city_name", "main.aqi"]].dropna()

# Ensure output folder exists
os.makedirs("arima_models", exist_ok=True)

# Get unique cities
cities = data["city_name"].unique()

for city in tqdm(cities, desc="Training ARIMA models"):
    try:
        # Filter data for the city
        city_data = data[data["city_name"] == city].copy()

        # Prepare time series
        city_data = city_data[["datetime", "main.aqi"]]
        city_data.set_index("datetime", inplace=True)

        # Resample to daily mean AQI only (main.aqi is numeric)
        city_data = city_data.resample("D").mean()

        # Interpolate missing values
        city_data["main.aqi"].interpolate(method="linear", inplace=True)

        # Fit ARIMA model
        model = ARIMA(city_data["main.aqi"], order=(2, 1, 2))
        model_fit = model.fit()

        # Save model
        city_filename = city.lower().replace(" ", "_")
        model_fit.save(f"arima_models/{city_filename}_arima.pkl")

    except Exception as e:
        print(f"[!] Skipping {city} due to error: {e}")
