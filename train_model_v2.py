import pandas as pd
import lightgbm as lgb
import pickle
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Added MAE
from sklearn.preprocessing import LabelEncoder

# Load processed dataset
df = pd.read_csv("final_dataset.csv")

# --- Preprocessing & Feature Engineering ---

# Label encode the city_name column
le = LabelEncoder()
df['city_name'] = le.fit_transform(df['city_name'])

# Save the fitted LabelEncoder for the forecasting app to use
joblib.dump(le, "city_label_encoder.pkl")

# Convert 'datetime' column to datetime format and create time features
df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['year'] = df['datetime'].dt.year
df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)


# --- Feature Selection ---
# CRITICAL CHANGE: We explicitly define the features to use.
# 'aqi_lag1' and 'aqi_lag2' are removed to force the model to learn
# from the pollutant components, making the forecast more dynamic.
features = [
    'city_name', # This is now the encoded version
    'coord.lat',
    'coord.lon',
    'components.co',
    'components.no',
    'components.no2',
    'components.o3',
    'components.so2',
    'components.pm2_5',
    'components.pm10',
    'components.nh3',
    'day_of_week',
    'month',
    'hour',
    'day',
    'year',
    'is_weekend'
]

# Target and features
target = "main.aqi"
X = df[features]
y = df[target]

# Define a date for splitting (make sure it's timezone-naive)
cutoff_date = pd.to_datetime("2024-12-25")

# Split data into train and test based on date
train_mask = df['datetime'] < cutoff_date
test_mask = df['datetime'] >= cutoff_date

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# LightGBM parameters (kept from your original script)
params = {
    "objective": "regression",
    "metric": ["rmse", "l1"], # Changed to track both RMSE and MAE (l1)
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.07,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42
}

# Train with early stopping as a callback
print("Training new AQI model without lag features...")
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_eval],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

# --- Save Artifacts ---

# Save the trained model
with open("aqi_forecasting_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the list of features that this model was trained on
with open("used_features.pkl", "wb") as f:
    pickle.dump(features, f)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred) # Added MAE calculation
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ New dynamic AQI model trained and saved as 'aqi_forecasting_model.pkl'")
print(f"   Encoder saved as 'city_label_encoder.pkl'")
print(f"   Feature list saved as 'used_features.pkl'")
print(f"\n--- Evaluation Metrics ---")
print(f"📉 Mean Absolute Error (MAE): {mae:.4f}") # Added MAE to output
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"📈 R-squared (R²): {r2:.4f}")
print("\nNote: A slightly lower R² score is expected and acceptable, as the model is now more dynamic.")
