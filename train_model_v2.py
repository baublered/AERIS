import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from feature_engineering import (
    create_lag_features,
    create_time_features,
    create_city_rolling_features,
    create_interaction_features
)
from data_preprocessing import load_and_clean_data

# === Step 1: Load and preprocess data ===
print("📦 Loading and preprocessing data...")
df = load_and_clean_data("cleaned_air_quality_data.csv")

# Filter out invalid PM2.5 values
df = df[(df['components.pm2_5'] > 0) & (df['components.pm2_5'] < 500)]

# Feature Engineering
df["datetime"] = pd.to_datetime(df["datetime"])
df = create_lag_features(df)
df = create_time_features(df)
df = create_city_rolling_features(df)
df = create_interaction_features(df)

# Drop any NA rows after feature engineering
df.dropna(inplace=True)

# === Step 2: Define features and target ===
target = "components.pm2_5"
features = [col for col in df.columns if col not in ["datetime", "city_name", "season", target]]

# Save used features for prediction later
joblib.dump(features, "used_features.pkl")

# === Step 3: Split into train, val, and test ===
print("🔪 Splitting into train/validation/test...")
train_df = df[df["year"] == 2023]
val_df = df[(df["year"] == 2024) & (df["month"] <= 6)]
test_df = df[(df["year"] == 2024) & (df["month"] > 6)]

X_train, y_train = train_df[features], np.log1p(train_df[target])
X_val, y_val = val_df[features], np.log1p(val_df[target])
X_test, y_test = test_df[features], np.log1p(test_df[target])

# === Step 4: Train LightGBM model ===
print("🚀 Training LightGBM model...")
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

callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=100)
]

model = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[train_data, val_data],
    callbacks=callbacks
)

# === Step 5: Evaluate on validation set ===
print("🧪 Evaluating on validation set...")
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
y_val_pred_inv = np.expm1(y_val_pred)
y_val_true = np.expm1(y_val)

mae_val = mean_absolute_error(y_val_true, y_val_pred_inv)
print(f"\n✅ Final MAE on 2024 Validation (Jan–Jun): {mae_val:.4f} µg/m³")

# === Step 6: Plot feature importance ===
print("\n📊 Plotting feature importances...")
lgb.plot_importance(model, max_num_features=20)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# === Step 7: Save the model ===
joblib.dump(model, "pm2_5_forecasting_model.pkl")
print("💾 Model saved as pm2_5_forecasting_model.pkl")

# === (Optional) Step 8: Save test data for later testing ===
joblib.dump((X_test, y_test), "test_data.pkl")
print("🛡️ Test set saved as test_data.pkl for future evaluation.")
