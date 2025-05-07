
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# Regression evaluation
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

# Classification evaluation
def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Plot feature importance
def plot_feature_importance(model):
    lgb.plot_importance(model, max_num_features=10)
    plt.show()

# Example usage:
# Assuming you have your true values (y_true) and predictions (y_pred)
# For regression:
# y_true = # actual PM2.5 values
# y_pred = # predicted PM2.5 values
# evaluate_regression(y_true, y_pred)

# For classification:
# y_true = # actual AQI categories
# y_pred = # predicted AQI categories
# evaluate_classification(y_true, y_pred)

# For plotting feature importance:
# model = # your trained LightGBM model
# plot_feature_importance(model)
