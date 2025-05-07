import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle

# Load data
df = pd.read_csv("cleaned_air_quality_data.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# Rename column for easier handling
df['pm2_5'] = df['components.pm2_5']

# Label encode city
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city_name'])

# Feature engineering (lags)
lags = [1, 2, 3, 4, 24]
for lag in lags:
    df[f'pm2_5_lag_{lag}h'] = df.groupby('city_name')['pm2_5'].shift(lag)

# Rolling features (based on shifted values to prevent leakage)
df['pm2_5_roll_mean_3h'] = df.groupby('city_name')['pm2_5'].shift(1).rolling(3).mean()
df['pm2_5_roll_median_6h'] = df.groupby('city_name')['pm2_5'].shift(1).rolling(6).median()
df['pm2_5_roll_min_6h'] = df.groupby('city_name')['pm2_5'].shift(1).rolling(6).min()
df['pm2_5_roll_max_6h'] = df.groupby('city_name')['pm2_5'].shift(1).rolling(6).max()
df['pm2_5_roll_std_6h'] = df.groupby('city_name')['pm2_5'].shift(1).rolling(6).std()

# Drop rows with NaNs from lag/rolling generation
df.dropna(inplace=True)

# Split train/test chronologically
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Define features
features = [
    'city_encoded', 'hour', 'day_of_week', 'month', 'is_weekend',
    'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_3h', 'pm2_5_lag_4h', 'pm2_5_lag_24h',
    'pm2_5_roll_mean_3h', 'pm2_5_roll_median_6h', 'pm2_5_roll_min_6h',
    'pm2_5_roll_max_6h', 'pm2_5_roll_std_6h'
]

# Prepare training data
X_train = train_df[features].copy()
y_train = train_df['pm2_5'].copy()

# Initialize LightGBM model
model = LGBMRegressor(n_estimators=100)

# --- 🧠 Teacher-forcing loop ---
teacher_forcing_ratio = 0.5
X_simulated = X_train.copy()

for i in range(5):  # multiple passes to simulate recursive lags
    model.fit(X_simulated, y_train)

    for lag_feature in [f'pm2_5_lag_{lag}h' for lag in lags]:
        if np.random.rand() > teacher_forcing_ratio:
            predictions = model.predict(X_simulated)
            X_simulated[lag_feature] = predictions  # replace lag with simulated value

# Final model fit with simulated lags
model.fit(X_simulated, y_train)

# Save model and artifacts
with open("pm2_5_model_recursive.pkl", "wb") as f:
    pickle.dump(model, f)

with open("used_features_recursive.pkl", "wb") as f:
    pickle.dump(features, f)

joblib.dump(le, "city_label_encoder.pkl")
