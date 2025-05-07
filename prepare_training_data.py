import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("cleaned_air_quality_data.csv", parse_dates=["datetime"])

# Sort by city and datetime
df = df.sort_values(by=["city_name", "datetime"])

# Select existing features
features = [
    "datetime", "city_name",
    "components.pm2_5", "components.pm10", "components.no2",
    "components.so2", "components.co", "components.o3", "components.no", "components.nh3",
    "day_of_week", "month", "is_weekend"
]

df = df[features]

# Drop missing values
df.dropna(inplace=True)

# Create lag features (for PM2.5)
df["pm2_5_lag1"] = df.groupby("city_name")["components.pm2_5"].shift(1)
df["pm2_5_lag2"] = df.groupby("city_name")["components.pm2_5"].shift(2)

# Drop rows with NaNs from lagging
df.dropna(inplace=True)

# Label encode city
df["city_name"] = LabelEncoder().fit_transform(df["city_name"])

# Define target and features
target = "components.pm2_5"
feature_cols = [
    "components.pm10", "components.no2", "components.so2", "components.co", 
    "components.o3", "components.no", "components.nh3", 
    "day_of_week", "month", "is_weekend", "pm2_5_lag1", "pm2_5_lag2", "city_name"
]

# Save used features
with open("used_features.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Save processed dataset
df[feature_cols + [target]].to_csv("processed_training_data.csv", index=False)

print("✅ Processed training data saved to 'processed_training_data.csv'")
