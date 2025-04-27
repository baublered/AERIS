#for model evaluation, we can use the validation set to tune hyperparameters and avoid overfitting.

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Step 1: Load model and test set ===
print("📦 Loading model and test data...")
model = joblib.load("pm2_5_forecasting_model.pkl")
X_test, y_test = joblib.load("test_data.pkl")

# === Step 2: Predict on test set ===
print("🚀 Predicting on Test Set...")
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Inverse the log1p transformation
y_test_pred_inv = np.expm1(y_test_pred)
y_test_true_inv = np.expm1(y_test)

# === Step 3: Evaluate ===
print("🧪 Evaluating model performance...")

# Mean Absolute Error (MAE)
mae_test = mean_absolute_error(y_test_true_inv, y_test_pred_inv)
print(f"✅ Final MAE on Test Set (Jul–Dec 2024): {mae_test:.4f} µg/m³")

# Root Mean Squared Error (RMSE)
rmse_test = np.sqrt(mean_squared_error(y_test_true_inv, y_test_pred_inv))
print(f"✅ Final RMSE on Test Set: {rmse_test:.4f} µg/m³")

# R² Score
r2 = r2_score(y_test_true_inv, y_test_pred_inv)
print(f"✅ R² Score on Test Set: {r2:.4f}")

# === Step 4: Save Test Results to CSV ===
print("💾 Saving test predictions to test_predictions.csv...")
test_results = pd.DataFrame({
    "True_PM2_5": y_test_true_inv,
    "Predicted_PM2_5": y_test_pred_inv
})
test_results.to_csv("test_predictions.csv", index=False)

# === Step 5: Plot True vs Predicted ===
print("📊 Plotting True vs Predicted Scatter Plot...")
plt.figure(figsize=(8, 6))
plt.scatter(y_test_true_inv, y_test_pred_inv, alpha=0.5)
plt.plot([0, max(y_test_true_inv.max(), y_test_pred_inv.max())],
        [0, max(y_test_true_inv.max(), y_test_pred_inv.max())],
        color="red", linestyle="--")
plt.xlabel("True PM2.5 (µg/m³)")
plt.ylabel("Predicted PM2.5 (µg/m³)")
plt.title("True vs Predicted PM2.5 on Test Set")
plt.grid(True)
plt.tight_layout()
plt.savefig("true_vs_predicted_scatter.png")
plt.show()

print("\n🎯 Test evaluation completed successfully!")
