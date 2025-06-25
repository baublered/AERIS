# eda_aeris.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # Added for directory creation

# --- Configuration ---
DATA_FILEPATH = "final_dataset.csv"
PLOT_OUTPUT_DIR = "eda_plots"

# --- Helper Functions ---
def setup_eda():
    """Sets up the environment for EDA."""
    sns.set(style="whitegrid")
    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
        print(f"Created directory for plots: {PLOT_OUTPUT_DIR}")
    return PLOT_OUTPUT_DIR

def load_data(filepath):
    """Loads and preprocesses the dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        # --- Preprocessing ---
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            initial_rows = len(df)
            df.dropna(subset=['datetime'], inplace=True)
            if len(df) < initial_rows:
                print(f"Dropped {initial_rows - len(df)} rows with invalid datetime values.")
        else:
            print("Warning: 'datetime' column not found.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None

# --- Core EDA Functions ---

def display_basic_info(df):
    """Displays basic information and descriptive statistics."""
    print("\n--- 1. Dataset Info & Description ---")
    print("--- Dataset Info ---")
    df.info()
    print("\n--- Dataset Description ---")
    print(df.describe())
    print("-" * 40)

def plot_correlation_heatmap(df, save_path):
    """Plots a heatmap of the correlation matrix for numeric features."""
    print("\n--- 2. Correlation Analysis ---")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # For clarity, let's remove some non-feature columns if they exist
    cols_to_exclude = ['year', 'month', 'day', 'hour']
    numeric_features = [col for col in numeric_cols if col not in cols_to_exclude]
    
    correlation_matrix = df[numeric_features].corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Correlation heatmap saved to {save_path}")
    print("-" * 40)

def plot_target_distribution(df, save_path):
    """Plots the distribution of the main AQI values."""
    print("\n--- 3. Target Variable (AQI) Distribution ---")
    if 'main.aqi' not in df.columns:
        print("Error: 'main.aqi' column not found.")
        return

    plt.figure(figsize=(10, 6))
    sns.countplot(x='main.aqi', data=df, palette='viridis', order=sorted(df['main.aqi'].unique()))
    plt.title('Distribution of Main AQI Values', fontsize=16)
    plt.xlabel('AQI Value (1=Good, 5=Very Poor)')
    plt.ylabel('Count')
    plt.savefig(save_path)
    plt.close()
    print(f"AQI distribution plot saved to {save_path}")
    print("-" * 40)

def plot_pollutant_distributions(df, save_path):
    """Plots the distributions of individual pollutant components."""
    print("\n--- 4. Pollutant Feature Distributions ---")
    pollutants = [
        'components.pm2_5', 'components.pm10', 'components.o3',
        'components.no2', 'components.so2', 'components.co'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distributions of Key Pollutants', fontsize=20)
    axes = axes.flatten()

    for i, pollutant in enumerate(pollutants):
        if pollutant in df.columns:
            sns.histplot(df[pollutant], ax=axes[i], kde=True, bins=50)
            axes[i].set_title(f'Distribution of {pollutant}')
            axes[i].set_xlabel('Concentration (µg/m³ or similar)')
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Pollutant distributions plot saved to {save_path}")
    print("-" * 40)

def plot_geospatial_aqi_distribution(df, save_path):
    """Plots the average AQI on a map based on city coordinates."""
    print("\n--- 5. Geospatial AQI Distribution ---")
    if not all(c in df.columns for c in ['coord.lon', 'coord.lat', 'main.aqi']):
        print("Error: Missing coordinate or AQI columns for geospatial plot.")
        return

    city_avg_aqi = df.groupby(['city_name', 'coord.lat', 'coord.lon'])['main.aqi'].mean().reset_index()

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=city_avg_aqi,
        x='coord.lon',
        y='coord.lat',
        hue='main.aqi',
        size='main.aqi',
        sizes=(50, 500),
        palette='viridis_r',
        alpha=0.8
    )
    plt.title('Geospatial Distribution of Average AQI', fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Average AQI')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Geospatial plot saved to {save_path}")
    print("-" * 40)

def plot_temporal_trends(df, save_path_prefix):
    """Plots various trends over time."""
    print("\n--- 6. Temporal Analysis ---")
    if 'datetime' not in df.columns:
        print("Error: 'datetime' column required for temporal analysis.")
        return

    # Trend 1: Records per year
    plt.figure(figsize=(10, 5))
    df['datetime'].dt.year.value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Number of Records per Year')
    plt.ylabel('Count')
    plt.savefig(f"{save_path_prefix}_records_per_year.png")
    plt.close()
    print(f"Records per year plot saved.")

    # Trend 2: Weekly Average AQI
    weekly_aqi = df.set_index('datetime')['main.aqi'].resample('W').mean()
    plt.figure(figsize=(15, 6))
    weekly_aqi.plot(title='Weekly Average AQI Trend Over Time', marker='.', linestyle='-')
    plt.ylabel('Average AQI')
    plt.savefig(f"{save_path_prefix}_weekly_aqi_trend.png")
    plt.close()
    print(f"Weekly AQI trend plot saved.")

    # Trend 3: Monthly Average AQI Heatmap (Top 10 Cities)
    top_cities = df['city_name'].value_counts().nlargest(10).index
    filtered_df = df[df['city_name'].isin(top_cities)]
    
    pivot_table = filtered_df.pivot_table(
        index=filtered_df['datetime'].dt.strftime('%b'),
        columns='city_name',
        values='main.aqi',
        aggfunc='mean'
    ).reindex(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5)
    plt.title('Monthly Average AQI by Top 10 Cities')
    plt.xlabel('City')
    plt.ylabel('Month')
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_monthly_heatmap_top_cities.png")
    plt.close()
    print(f"Monthly heatmap saved.")
    print("-" * 40)


if __name__ == "__main__":
    # --- Main Analysis Flow ---
    output_dir = setup_eda()
    df = load_data(DATA_FILEPATH)

    if df is not None:
        # 1. Basic Information
        display_basic_info(df)

        # 2. Correlation Analysis
        plot_correlation_heatmap(df, os.path.join(output_dir, "correlation_heatmap.png"))

        # 3. Target Variable Analysis
        plot_target_distribution(df, os.path.join(output_dir, "aqi_distribution.png"))

        # 4. Feature Analysis (Pollutants)
        plot_pollutant_distributions(df, os.path.join(output_dir, "pollutant_distributions.png"))

        # 5. Geospatial Analysis
        plot_geospatial_aqi_distribution(df, os.path.join(output_dir, "geospatial_aqi_map.png"))

        # 6. Temporal Analysis (Time-based)
        plot_temporal_trends(df, os.path.join(output_dir, "temporal"))

        print("\n✅ Exploratory Data Analysis complete. All plots saved in the 'eda_plots' directory.")
    else:
        print("Data loading failed. Exiting EDA.")