import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and model
df = pd.read_csv("final_dataset.csv")

# Label encode 'city_name'
le = LabelEncoder()
df['city_name'] = le.fit_transform(df['city_name'])

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

# Load model and features
with open("aqi_forecasting_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("used_features.pkl", "rb") as f:
    features = pickle.load(f)

# Set target and features
target = "main.aqi"
X = df[features]
y = df[target]

# Split data by datetime
cutoff_date = pd.to_datetime("2024-12-25")
train_mask = df["datetime"] < cutoff_date
test_mask = df["datetime"] >= cutoff_date

X_test = X[test_mask]
y_test = y[test_mask]

# Predict
y_pred = model.predict(X_test)

# Round predictions to get AQI categories (1–5)
y_pred_rounded = np.clip(np.round(y_pred), 1, 5).astype(int)

# Metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_rounded)
conf_matrix = confusion_matrix(y_test, y_pred_rounded)
report = classification_report(y_test, y_pred_rounded)

# Print results
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 R² Score: {r2:.2f}")
print(f"✅ Classification Accuracy: {accuracy:.2%}")
print("\n📊 Classification Report:\n", report)
print("\n🧩 Confusion Matrix:\n", conf_matrix)

#---- for residuals ----
# Compute residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color="royalblue")

# Zero line
plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Perfect Prediction (Residual = 0)')

# Labels and title
plt.title("Residual Distribution of AQI Predictions", fontsize=16)
plt.xlabel("Residual (True AQI - Predicted AQI)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()