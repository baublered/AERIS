import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import joblib
import os

# --- Configuration ---
# This script uses your main dataset and pre-existing city encoder to train a PM2.5 model.
# It is designed to be run AFTER you have trained your main AQI model,
# as it relies on the 'city_label_encoder.pkl' created during that process.
DATA_FILE = 'final_dataset.csv'
ENCODER_INPUT_FILE = 'city_label_encoder.pkl' # Using the existing encoder
MODEL_OUTPUT_FILE = 'pm25_forecasting_model.pkl'
FEATURES_OUTPUT_FILE = 'pm25_used_features.pkl'
TARGET_COLUMN = 'components.pm2_5'

# --- Feature Engineering Functions ---
def create_time_features(df):
    """Creates time-based features from a datetime column."""
    # The 'datetime' column is expected to exist in the CSV.
    if 'datetime' not in df.columns:
        raise KeyError("The input DataFrame must contain a 'datetime' column.")
    
    # Convert the 'datetime' column to datetime objects if it's not already.
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
        
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    df['year'] = df['datetime'].dt.year
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    return df

def create_lag_features(df, group_col, target_col, lags):
    """Creates lag features for a target column, grouped by another column."""
    df = df.sort_values(by=[group_col, 'datetime'])
    for lag in lags:
        # Use a unique name for the lag feature to avoid conflicts
        df[f'pm25_lag{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    return df

# --- Main Training Script ---
def train_pm25_model():
    """Loads data, engineers features, trains a PM2.5 model, and saves artifacts."""
    # 1. Load Data and Encoder
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at '{DATA_FILE}'.")
        return
    if not os.path.exists(ENCODER_INPUT_FILE):
        print(f"Error: City encoder not found at '{ENCODER_INPUT_FILE}'.")
        print("Please train your main AQI model first to generate the encoder file.")
        return

    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loading city encoder from {ENCODER_INPUT_FILE}...")
    city_encoder = joblib.load(ENCODER_INPUT_FILE)

    # 2. Preprocessing and Feature Engineering
    print("Starting feature engineering...")
    
    # Ensure target column is numeric and handle missing values
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    df = create_time_features(df)

    # Use the pre-loaded encoder to transform city names
    # Filter out any cities from the dataset that are not in the encoder
    known_cities = city_encoder.classes_
    original_rows = len(df)
    df = df[df['city_name'].isin(known_cities)].copy()
    if len(df) < original_rows:
        print(f"Warning: Dropped {original_rows - len(df)} rows for cities not found in the encoder.")
    
    df['city_encoded'] = city_encoder.transform(df['city_name'])
    
    # Create lag features for PM2.5
    df = create_lag_features(df, 'city_name', TARGET_COLUMN, lags=[1, 2])
    
    # Drop rows with NaN values created by the lag operation
    df.dropna(inplace=True)
    if df.empty:
        print("Error: No data left after feature engineering. Dataset might be too small for lags.")
        return
    print("Feature engineering complete.")

    # 3. Define Features (X) and Target (y)
    # These features are chosen to match what the forecasting app can provide
    features = [
        'coord.lat', 'coord.lon', 'city_encoded',
        'day_of_week', 'month', 'is_weekend', 'year', 'day', 'hour',
        'pm25_lag1', 'pm25_lag2'
    ]
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: The following required features are missing from the dataset: {missing_features}")
        return

    X = df[features]
    y = df[TARGET_COLUMN]

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")

    # 5. Train LightGBM Model
    print("Training LightGBM model for PM2.5...")
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', # MAE is a good metric for this kind of data
        n_estimators=10000, #increased number of trees for better performance
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             eval_metric='l1',
             callbacks=[lgb.early_stopping(100, verbose=True)])

    # 6. Evaluate Model
    print("\nEvaluating model performance on the test set...")
    y_pred = lgbm.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R²): {r2:.4f}")

    # 7. Save Artifacts
    # We save the booster object for faster prediction
    with open(MODEL_OUTPUT_FILE, 'wb') as f:
        pickle.dump(lgbm.booster_, f)
    print(f"\nModel saved to {MODEL_OUTPUT_FILE}")

    # Save the list of features the model was trained on
    with open(FEATURES_OUTPUT_FILE, 'wb') as f:
        pickle.dump(features, f)
    print(f"Feature list saved to {FEATURES_OUTPUT_FILE}")
    print("\nTraining process complete.")

if __name__ == '__main__':
    train_pm25_model()