from data_preprocessing import load_and_clean_data
from feature_engineering import create_lag_features
from train import train_lightgbm
from visualize import plot_results
from model import save_model
import joblib

# Load and preprocess data
df = load_and_clean_data("cleaned_air_quality_data.csv")

# Ensure 'year' column exists
df["year"] = df["datetime"].dt.year

# Feature engineering
df = create_lag_features(df)

# Split into train and test sets
train_df = df[df['year'] < 2024]
test_df = df[df['year'] == 2024]

# Drop target and unwanted columns
X_train = train_df.drop(columns=['components.pm2_5', 'city_name', 'datetime'])
X_test = test_df.drop(columns=['components.pm2_5', 'city_name', 'datetime'])

y_train = train_df['components.pm2_5']
y_test = test_df['components.pm2_5']

# Save the feature names used for training
features = X_train.columns.tolist()
joblib.dump(features, "used_features.pkl")

# Train model
model, y_pred = train_lightgbm(X_train, y_train, X_test, y_test)

# Plot results
plot_results(test_df, y_test, y_pred)

# Save model
save_model(model)