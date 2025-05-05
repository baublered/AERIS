import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.sort_values("datetime")

    # Set datetime as index for time-based interpolation
    df.set_index('datetime', inplace=True)

    # Convert object columns to appropriate types before interpolation
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'city_name':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    # Improved missing value imputation: interpolate numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    # Reset index to bring datetime back as a column
    df.reset_index(inplace=True)

    # Optional: Outlier capping for pm2_5 (e.g., cap at 99th percentile)
    upper_cap = df['components.pm2_5'].quantile(0.99)
    df['components.pm2_5'] = df['components.pm2_5'].clip(upper=upper_cap)

    # Label encode city_name
    le = LabelEncoder()
    df["city_encoded"] = le.fit_transform(df["city_name"])

    # Save encoder for later use
    joblib.dump(le, "city_label_encoder.pkl")

    return df
def get_city_coordinates(city_name, coord_df):
    match = coord_df[coord_df['city_name'].str.lower() == city_name.lower()]
    if not match.empty:
        return match.iloc[0]['latitude'], match.iloc[0]['longitude']
    else:
        raise ValueError(f"City '{city_name}' not found in coordinates file.")
