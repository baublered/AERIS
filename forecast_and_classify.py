import pandas as pd
import joblib
from feature_engineering import create_lag_features
from model_utils import classify_pm2_5
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from evaluate import evaluate_model
import difflib
from feature_engineering import (
    create_lag_features,
    create_time_features,
    create_city_rolling_features
)

# === Step 1: Load cleaned and processed data ===
df = pd.read_csv("cleaned_air_quality_data.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")
df = create_lag_features(df)
df = create_time_features(df)
df = create_city_rolling_features(df)

# === Step 2: Load label encoder and list of valid cities ===
le = joblib.load("city_label_encoder.pkl")
cities = le.classes_

# === Step 3: User input and city matching ===
city = input("Enter a Philippine city name: ").strip()
matches = difflib.get_close_matches(city, cities, n=1, cutoff=0.6)

if city not in cities:
    if matches:
        print(f"Did you mean '{matches[0]}'?")
        city = matches[0]
    else:
        print("City not recognized. Please try another.")
        exit()

# === Step 4: Filter data for that city before 2025 ===
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
cutoff_date = pd.Timestamp("2025-01-01", tz="Asia/Manila").tz_convert("UTC")
city_df = df[(df["city_name"] == city) & (df["datetime"] < cutoff_date)]

if city_df.empty:
    print("No data available for that city.")
    exit()

# === Step 5: Prepare the latest row for prediction ===
latest_row = city_df.iloc[-1:].copy()

# Add encoded city column as it's required by the model
latest_row["city_encoded"] = le.transform([city])[0]

# Load the features used during training
features = joblib.load("used_features.pkl")
X_latest = latest_row[features]

# === Step 6: Load trained model ===
model = joblib.load("pm2_5_forecasting_model.pkl")

# === Step 7: Predict and classify ===
pm2_5_pred = model.predict(X_latest)[0]
aqi_category = classify_pm2_5(pm2_5_pred)

# === Step 8: Evaluate model on 2024 test data for that city ===
test_city_df = df[(df["year"] == 2024) & (df["city_name"] == city)].copy()
test_city_df["city_encoded"] = le.transform(test_city_df["city_name"])

X_test_city = test_city_df[features]
y_test_city = test_city_df['components.pm2_5']
y_pred_city = model.predict(X_test_city)

mae = mean_absolute_error(y_test_city, y_pred_city)
rmse = mean_squared_error(y_test_city, y_pred_city)


# === Step 9: Display the result ===
print("\n🌆 City:", city)
print(f"🔮 Predicted PM2.5: {pm2_5_pred:.2f} µg/m³")
print(f"🟡 Air Quality Category: {aqi_category}")
print(f"📊 MAE (2024): {mae:.2f}, RMSE: {rmse:.2f}\n")
print("📅 Date of Prediction:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

