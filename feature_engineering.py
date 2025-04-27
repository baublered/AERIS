def create_lag_features(df):
    # More lag features (up to 3 days)
    lag_hours = [1, 2, 3, 4, 6, 8, 12, 24, 36, 48, 72]
    for lag in lag_hours:
        df[f'pm2_5_lag_{lag}h'] = df['components.pm2_5'].shift(lag)

    # Rolling averages + std devs + median, min, max (across whole dataset)
    df['pm2_5_roll_mean_3h'] = df['components.pm2_5'].rolling(window=3).mean()
    df['pm2_5_roll_mean_6h'] = df['components.pm2_5'].rolling(window=6).mean()
    df['pm2_5_roll_std_6h'] = df['components.pm2_5'].rolling(window=6).std()
    df['pm2_5_roll_median_6h'] = df['components.pm2_5'].rolling(window=6).median()
    df['pm2_5_roll_min_6h'] = df['components.pm2_5'].rolling(window=6).min()
    df['pm2_5_roll_max_6h'] = df['components.pm2_5'].rolling(window=6).max()

    df.dropna(inplace=True)  # remove rows with NaN from lag/rolling
    return df


def create_time_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Add Philippine season feature (Dry: Nov-May, Wet: Jun-Oct), Tropical country
    df["month"] = df["datetime"].dt.month
    df["season"] = df["month"].apply(lambda x: "dry" if x in [11, 12, 1, 2, 3, 4, 5] else "wet")
    df["season_encoded"] = df["season"].map({"dry": 0, "wet": 1})

    return df


def create_city_rolling_features(df, window_sizes=[3, 6]):
    for window in window_sizes:
        df[f"pm2_5_city_roll_mean_{window}h"] = (
            df.groupby("city_name")["components.pm2_5"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df[f"pm2_5_city_roll_std_{window}h"] = (
            df.groupby("city_name")["components.pm2_5"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
    return df


def create_interaction_features(df):
    # Example interaction features: product and ratio of lag features
    if 'pm2_5_lag_1h' in df.columns and 'pm2_5_lag_3h' in df.columns:
        df['lag_1h_x_lag_3h'] = df['pm2_5_lag_1h'] * df['pm2_5_lag_3h']
        df['lag_1h_div_lag_3h'] = df['pm2_5_lag_1h'] / (df['pm2_5_lag_3h'] + 1e-5)
    return df
