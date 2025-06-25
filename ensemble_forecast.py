def predict_lightgbm(city):
    result = forecast_aqi_logic(city)
    if isinstance(result, dict) and "error" in result:
        return {"error": result["error"]}
    forecast_data, mae, rmse, r2, latest_aqi_api_val = result
    return {
        "model": "LightGBM",
        "forecast": forecast_data,
        "latest_aqi": latest_aqi_api_val,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
    }
