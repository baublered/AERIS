# train_model_daily.py — Train with Features for Daily Forecast
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Assuming these scripts exist and contain the necessary functions
from feature_engineering import (
    create_lag_features,
    create_time_features,
    create_city_rolling_features, # Keep to ensure city_encoded is created (adjust if city encoding is elsewhere)
    create_interaction_features # Keep for completeness, though features might not be selected
)
from data_preprocessing import load_and_clean_data # Assuming this also handles city encoding or it's done in feature_engineering

# === Step 1: Load and preprocess data ===
print("📦 Loading and preprocessing data...")
# Assuming load_and_clean_data and subsequent steps create 'year' and 'city_encoded' columns
df = load_and_clean_data("cleaned_air_quality_data.csv")
df = df[(df['components.pm2_5'] > 0) & (df['components.pm2_5'] < 500)]
df["datetime"] = pd.to_datetime(df["datetime"])

# === Step 2: Feature engineering ===
# Generate all potential features first from your existing functions
df = create_lag_features(df) # Should create pm2_5_lag_24h and other lags
df = create_time_features(df) # Should create hour, dayofweek, month, is_weekend
# Keep rolling and interaction features for now, even if not selected, as they might be dependencies
df = create_city_rolling_features(df) # Should ideally create 'city_encoded' column
df = create_interaction_features(df)

# Ensure the target variable column name is correct
target = "components.pm2_5"

# Drop rows with NaN values after feature engineering (important after creating lags)
df.dropna(inplace=True)

# === Step 3: Define and select features for daily forecasting ===
# Choose features that are more suitable for daily forecasting and can be
# reasonably provided to the forecasting script (aeris_forecast.py) without
# requiring a full historical hourly dataset during inference.
selected_daily_features = [
    'month',
    'dayofweek',
    'is_weekend',
    'hour',           # Keep hour, assuming daily forecast is for a specific hour of the day
    'city_encoded',   # Assuming this feature is created during preprocessing/feature engineering
    'pm2_5_lag_24h'   # Represents the value 24 hours prior - can be approximated in forecasting
    # We exclude hourly rolling features and other specific lags as they are hard to recreate without history.
    # We also exclude the 'pm2_5' feature itself as a direct input feature,
    # as pm2_5_lag_24h serves as the historical context from the previous day.
]

print(f"📊 Selected features for daily model: {selected_daily_features}")

# Ensure all selected features exist in the DataFrame after feature engineering
for feature in selected_daily_features:
    if feature not in df.columns:
        print(f"Error: Selected feature '{feature}' not found in the DataFrame after feature engineering.")
        print("Available columns:", df.columns.tolist())
        print("\nPlease check your 'feature_engineering.py' and 'data_preprocessing.py' scripts to ensure these features are being created.")
        exit()

# Save the list of selected features for use in the forecasting script
used_features_filename = "used_features_daily.pkl"
joblib.dump(selected_daily_features, used_features_filename)
print(f"💾 Selected features saved as {used_features_filename}")


# === Step 4: Train/val/test split ===
print("🔪 Splitting into train/validation/test...")
# Use the same split logic as before
# Assuming 'year' column exists after loading/preprocessing
train_df = df[df["year"] == 2023]
val_df = df[(df["year"] == 2024) & (df["month"] <= 6)]
test_df = df[(df["year"] == 2024) & (df["month"] > 6)]

# Select only the selected daily features for training and evaluation
X_train, y_train = train_df[selected_daily_features], np.log1p(train_df[target])
X_val, y_val = val_df[selected_daily_features], np.log1p(val_df[target])
X_test, y_test = test_df[selected_daily_features], np.log1p(test_df[target])

# === Step 5: Train LightGBM ===
print("🚀 Training LightGBM model...")
# Use the same LightGBM parameters as before
params = {
    "objective": "regression",
    "learning_rate": 0.005,
    "num_leaves": 40,
    "max_depth": 12,
    "min_data_in_leaf": 15,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "metric": "l1"
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[train_data, val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)

# === Step 6: Evaluate ===
print("🧪 Evaluating on validation set...")
# Evaluate on validation set (using the original scale for MAE)
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
mae_val = mean_absolute_error(np.expm1(y_val), np.expm1(y_val_pred))
print(f"\n✅ MAE (Validation Jan–Jun 2024): {mae_val:.4f} µg/m³")

# === Step 7: Feature Importance ===
print("📊 Plotting Feature Importance...")
# Plot importance for the selected daily features
if selected_daily_features: # Ensure there are features to plot
    lgb.plot_importance(model, max_num_features=len(selected_daily_features))
    plt.title("PM2.5 Feature Importances (Selected Daily Features)")
    plt.tight_layout()
    plt.show()
else:
    print("No features selected for plotting importance.")


# === Step 8: Save model and test set ===
model_filename = "pm2_5_model_daily.pkl"
joblib.dump(model, model_filename)
print(f"💾 Model saved as {model_filename}")

test_data_filename = "test_data_daily.pkl"
joblib.dump((X_test, y_test), test_data_filename)
print(f"🛡️ Test set saved as {test_data_filename}")