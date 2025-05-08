import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load processed dataset
df = pd.read_csv("final_dataset.csv")

# Label encode the city_name column
le = LabelEncoder()
df['city_name'] = le.fit_transform(df['city_name'])

# Convert 'datetime' column to datetime format (remove timezone if exists)
df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)  # Remove timezone info

# Load used features
with open("used_features.pkl", "rb") as f:
    features = pickle.load(f)

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

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
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
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_eval],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

# Save model
with open("aqi_forecasting_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained and saved as 'aqi_forecasting_model.pkl'")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 R² Score: {r2:.2f}")
