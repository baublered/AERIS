import joblib

def save_model(model, filename="pm2_5_forecasting_model.pkl"):
    joblib.dump(model, filename)

def load_model(filename="pm2_5_forecasting_model.pkl"):
    return joblib.load(filename)