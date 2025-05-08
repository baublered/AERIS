import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load predictions
df = pd.read_csv("actual_vs_predicted.csv")

# Evaluation Metrics
mae = mean_absolute_error(df["actual_aqi"], df["predicted_aqi"])
rmse = mean_squared_error(df["actual_aqi"], df["predicted_aqi"], squared=False)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 1. Line Plot: Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(df["actual_aqi"].values, label="Actual AQI", marker='o')
plt.plot(df["predicted_aqi"].values, label="Predicted AQI", marker='x')
plt.title("Actual vs Predicted AQI")
plt.xlabel("Sample")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Scatter Plot: Predicted vs Actual
plt.figure(figsize=(6, 6))
sns.scatterplot(x="actual_aqi", y="predicted_aqi", data=df)
plt.plot([df["actual_aqi"].min(), df["actual_aqi"].max()],
         [df["actual_aqi"].min(), df["actual_aqi"].max()], 'r--')
plt.title("Predicted vs Actual AQI")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.tight_layout()
plt.show()

# 3. Distribution of Prediction Errors
df["error"] = df["predicted_aqi"] - df["actual_aqi"]
plt.figure(figsize=(8, 4))
sns.histplot(df["error"], bins=20, kde=True)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.tight_layout()
plt.show()
