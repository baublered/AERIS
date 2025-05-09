import pickle
import matplotlib.pyplot as plt
import lightgbm as lgb

# Load used features
with open("used_features.pkl", "rb") as f:
    used_features = pickle.load(f)

# Load trained LightGBM model
with open("aqi_forecasting_model.pkl", "rb") as f:
    model = pickle.load(f)

# Get feature importances
importances = model.feature_importance()
feature_importance_dict = dict(zip(used_features, importances))

# Sort by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Plotting
plt.figure(figsize=(12, 6))
plt.bar([x[0] for x in sorted_features], [x[1] for x in sorted_features], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Feature Importance in AQI Forecasting Model")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
