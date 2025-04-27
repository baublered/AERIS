def classify_pm2_5(pm_value):
    if pm_value <= 12.0:
        return "Good"
    elif pm_value <= 35.4:
        return "Moderate"
    elif pm_value <= 55.4:
        return "Poor"
    else:
        return "Hazardous"