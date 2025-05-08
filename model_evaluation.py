import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and features
with open("aqi_forecasting_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("used_features.pkl", "rb") as f:
    used_features = pickle.load(f)

# Load and prepare data
df = pd.read_csv("final_dataset.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# Label encode city_name to match training
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['city_name'] = le.fit_transform(df['city_name'])

# Split into train/test
train_df = df[df['datetime'] < "2024-12-25"]
test_df = df[df['datetime'] >= "2024-12-25"]

X_test = test_df[used_features]
y_test = test_df['main.aqi']

# Predict AQI
y_pred = model.predict(X_test)

# Regression metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("📊 Regression Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 🔍 Plot 1: Actual vs Predicted
plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.tight_layout()
plt.show()

# 🔍 Plot 2: Residuals vs Predicted
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted AQI")
plt.tight_layout()
plt.show()

# AQI category classification (1-5 scale)
def classify_aqi(aqi):
    if aqi == 1:
        return "Good"
    elif aqi == 2:
        return "Fair"
    elif aqi == 3:
        return "Moderate"
    elif aqi == 4:
        return "Poor"
    else:
        return "Very Poor"

# Convert predicted values to AQI categories
y_true_class = y_test.apply(classify_aqi)
y_pred_class = pd.Series(np.round(y_pred)).apply(lambda x: classify_aqi(int(x)))

# Classification metrics
aqi_accuracy = accuracy_score(y_true_class, y_pred_class)
print("\n🏷️ AQI Category Classification:")
print(f"Accuracy: {aqi_accuracy * 100:.2f}%")

# Confusion matrix
labels = ["Good", "Fair", "Moderate", "Poor", "Very Poor"]
cm = confusion_matrix(y_true_class, y_pred_class, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("AQI Category Confusion Matrix")
plt.tight_layout()
plt.show()
