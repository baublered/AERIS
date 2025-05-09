# eda_aeris.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np

def load_data(filepath):
    """Loads the dataset from a specified CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def display_basic_info(df):
    """Displays basic information and descriptive statistics about the DataFrame."""
    print("\n--- Dataset Info ---")
    df.info()
    print("\n--- Dataset Description ---")
    print(df.describe())
    print("-" * 20)

def preprocess_data(df):
    """Performs necessary data preprocessing steps."""
    # Convert 'datetime' to datetime format, coercing errors
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        print("\n'datetime' column converted to datetime objects.")
        # Drop rows where datetime conversion failed (NaT)
        initial_rows = len(df)
        df.dropna(subset=['datetime'], inplace=True)
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows with invalid datetime values.")
    else:
        print("\nWarning: 'datetime' column not found.")
    return df

def plot_aqi_distribution(df):
    """Plots the distribution of the main AQI values."""
    if 'main.aqi' not in df.columns:
        print("Error: 'main.aqi' column not found for distribution plot.")
        return

    # Drop rows where 'main.aqi' is NaN for counting
    aqi_counts = df['main.aqi'].dropna().value_counts().sort_index()

    if aqi_counts.empty:
        print("Warning: No valid 'main.aqi' data for distribution plot.")
        print("-" * 20)
        return

    print("\n--- Main AQI Distribution ---")
    for value, count in aqi_counts.items():
        print(f"AQI {value} = {count}")

    plt.figure(figsize=(10, 5))
    sns.barplot(x=aqi_counts.index, y=aqi_counts.values, color="skyblue", edgecolor="black")
    plt.title('Main AQI Distribution')
    plt.xlabel('AQI Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    print("-" * 20)

def plot_records_per_year(df):
    """Plots the number of records per year."""
    if 'datetime' not in df.columns or df['datetime'].empty:
        print("Error: 'datetime' column not found or is empty for records per year plot.")
        return

    # Drop rows where 'datetime' is NaT after coercion
    year_counts = df['datetime'].dropna().dt.year.value_counts().sort_index()

    if year_counts.empty:
        print("Warning: No valid datetime data for records per year plot.")
        print("-" * 20)
        return

    print("\n--- Record Count per Year ---")
    for year, count in year_counts.items():
        print(f"{year} = {count}")

    plt.figure(figsize=(10, 5))
    sns.barplot(x=year_counts.index, y=year_counts.values, color="lightgreen", edgecolor="black")
    plt.title('Number of Records per Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    print("-" * 20)

def plot_aqi_trend_top_cities(df, n_cities=5):
    """Plots the average AQI trend over time for the top N cities by record count."""
    if 'city_name' not in df.columns or 'datetime' not in df.columns or 'main.aqi' not in df.columns:
        print("Error: Required columns (city_name, datetime, main.aqi) not found for AQI trend plot.")
        return

    # Drop rows with missing essential data before finding top cities
    temp_df = df.dropna(subset=['city_name', 'datetime', 'main.aqi']).copy()

    if temp_df.empty:
        print("Warning: No valid data after dropping NaNs for AQI trend plot.")
        print("-" * 20)
        return

    top_cities = temp_df['city_name'].value_counts().nlargest(n_cities).index

    if top_cities.empty:
        print("Warning: Could not find top cities after dropping NaNs for AQI trend plot.")
        print("-" * 20)
        return

    print(f"\n--- AQI Trend for Top {n_cities} Cities ---")
    print(f"Top {n_cities} cities: {list(top_cities)}")

    filtered_df = temp_df[temp_df['city_name'].isin(top_cities)].copy()

    aqi_trend = filtered_df.groupby(['datetime', 'city_name'])['main.aqi'].mean().reset_index()

    if aqi_trend.empty:
        print("Warning: Grouping resulted in an empty DataFrame for AQI trend plot.")
        print("-" * 20)
        return

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=aqi_trend, x='datetime', y='main.aqi', hue='city_name', linewidth=1.5, alpha=0.9)
    plt.title(f'AQI Trends Over Time (Top {n_cities} Cities)')
    plt.xlabel('Date')
    plt.ylabel('Average AQI')
    plt.legend(title='City', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()
    print("-" * 20)

def plot_aqi_boxplot_top_cities(df, n_cities=5):
    """Creates a boxplot of AQI distribution for the top N cities."""
    if 'city_name' not in df.columns or 'main.aqi' not in df.columns:
        print("Error: Required columns (city_name, main.aqi) not found for AQI boxplot.")
        return

    # Drop rows with missing essential data before finding top cities
    temp_df = df.dropna(subset=['city_name', 'main.aqi']).copy()

    if temp_df.empty:
        print("Warning: No valid data after dropping NaNs for AQI boxplot.")
        print("-" * 20)
        return

    top_cities = temp_df['city_name'].value_counts().nlargest(n_cities).index

    if top_cities.empty:
        print("Warning: Could not find top cities after dropping NaNs for AQI boxplot.")
        print("-" * 20)
        return

    print(f"\n--- AQI Boxplot for Top {n_cities} Cities ---")
    filtered_df = temp_df[temp_df['city_name'].isin(top_cities)].copy()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=filtered_df, x='city_name', y='main.aqi', palette='pastel')
    plt.title(f'AQI Distribution by Top {n_cities} Cities')
    plt.xlabel('City')
    plt.ylabel('Main AQI')
    plt.tight_layout()
    plt.show()
    print("-" * 20)


def plot_monthly_aqi_heatmap_top_cities(df, n_cities=5):
    """Creates a heatmap of monthly average AQI for the top N cities."""
    if 'city_name' not in df.columns or 'datetime' not in df.columns or 'main.aqi' not in df.columns:
        print("Error: Required columns (city_name, datetime, main.aqi) not found for monthly heatmap.")
        return

    # Drop rows with missing essential data before finding top cities
    temp_df = df.dropna(subset=['city_name', 'datetime', 'main.aqi']).copy()

    if temp_df.empty:
        print("Warning: No valid data after dropping NaNs for monthly heatmap.")
        print("-" * 20)
        return

    top_cities = temp_df['city_name'].value_counts().nlargest(n_cities).index

    if top_cities.empty:
        print("Warning: Could not find top cities after dropping NaNs for monthly heatmap.")
        print("-" * 20)
        return

    print(f"\n--- Monthly Average AQI Heatmap for Top {n_cities} Cities ---")
    filtered_df = temp_df[temp_df['city_name'].isin(top_cities)].copy()

    # Ensure datetime is processed and has no NaT values
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['datetime']):
         filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'], errors='coerce')
         filtered_df.dropna(subset=['datetime'], inplace=True)

    if filtered_df.empty:
         print("Warning: Filtered DataFrame is empty after ensuring valid datetimes. Skipping heatmap.")
         print("-" * 20)
         return

    filtered_df['month'] = filtered_df['datetime'].dt.strftime('%b')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    pivot_table = filtered_df.pivot_table(
        index='month',
        columns='city_name',
        values='main.aqi',
        aggfunc='mean'
    ).reindex(month_order)

    if pivot_table.empty:
         print("Warning: Pivot table is empty after filtering. Skipping heatmap.")
         print("-" * 20)
         return

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5)
    plt.title(f'Monthly Average AQI by Top {n_cities} Cities')
    plt.xlabel('City')
    plt.ylabel('Month')
    plt.tight_layout()
    plt.show()
    print("-" * 20)


def plot_city_date_heatmap(df, start_date=None, end_date=None, n_cities=None):
    """
    Creates a heatmap of AQI across cities and dates, with optional filtering.

    Args:
        df (pd.DataFrame): The input DataFrame.
        start_date (str or datetime.date or pandas.Timestamp, optional): The start date for filtering
                                                (inclusive). Defaults to None.
                                                Can be a string ('YYYY-MM-DD'), date object, or Timestamp.
        end_date (str or datetime.date or pandas.Timestamp, optional): The end date for filtering
                                              (inclusive). Defaults to None.
                                              Can be a string ('YYYY-MM-DD'), date object, or Timestamp.
        n_cities (int, optional): The number of top cities to include
                                  (based on record count *within the filtered date range*).
                                  Defaults to None (include all cities).
    """
    if 'city_name' not in df.columns or 'datetime' not in df.columns or 'main.aqi' not in df.columns:
         print("Error: Required columns (city_name, datetime, main.aqi) not found for city-date heatmap.")
         return

    print("\n--- City vs Date AQI Heatmap ---")

    filtered_df = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # Ensure datetime is processed and has no NaT values before any filtering
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['datetime']):
         filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'], errors='coerce')
         filtered_df.dropna(subset=['datetime'], inplace=True)
    else:
        # If already datetime, ensure no NaT values exist
        filtered_df.dropna(subset=['datetime'], inplace=True)


    if filtered_df.empty:
        print("Warning: DataFrame is empty or has no valid datetimes after initial processing. Skipping heatmap.")
        print("-" * 20)
        return

    # --- Apply Date Range Filter ---
    if start_date is not None or end_date is not None:
        print(f"Filtering dates between {start_date if start_date else 'beginning'} and {end_date if end_date else 'end'}...")

        # Convert start_date and end_date parameters to date objects for comparison
        start_date_dt = None
        if isinstance(start_date, pd.Timestamp):
            start_date_dt = start_date.date()
        elif isinstance(start_date, str):
             try:
                 start_date_dt = pd.to_datetime(start_date).date()
             except ValueError:
                 print(f"Warning: Invalid start_date string format: {start_date}. Ignoring start date filter.")
        elif isinstance(start_date, datetime.date):
            start_date_dt = start_date


        end_date_dt = None
        if isinstance(end_date, pd.Timestamp):
            end_date_dt = end_date.date()
        elif isinstance(end_date, str):
             try:
                 end_date_dt = pd.to_datetime(end_date).date()
             except ValueError:
                 print(f"Warning: Invalid end_date string format: {end_date}. Ignoring end date filter.")
        elif isinstance(end_date, datetime.date):
            end_date_dt = end_date


        # Perform comparison using the .dt.date accessor to compare date objects
        if start_date_dt is not None:
            filtered_df = filtered_df[filtered_df['datetime'].dt.date >= start_date_dt]
        if end_date_dt is not None:
            filtered_df = filtered_df[filtered_df['datetime'].dt.date <= end_date_dt]


        if filtered_df.empty:
            print("Warning: Date filtering resulted in an empty DataFrame. Skipping heatmap.")
            print("-" * 20)
            return
        else:
            print(f"Data filtered from {filtered_df['datetime'].min().date()} to {filtered_df['datetime'].max().date()}.")


    # --- Apply Top N Cities Filter ---
    if n_cities is not None and n_cities > 0:
        # Find top cities *within the filtered date range*
        if 'city_name' in filtered_df.columns:
            # Ensure there's data left after date filtering before finding top cities
            if not filtered_df.empty:
                 # Drop NaNs in 'city_name' for accurate value_counts
                 top_cities_series = filtered_df['city_name'].dropna().value_counts().nlargest(n_cities)

                 if top_cities_series.empty:
                     print(f"Warning: No valid city names found in the filtered data to determine top {n_cities} cities.")
                     filtered_df = pd.DataFrame() # Set to empty to skip heatmap
                 else:
                     top_cities = top_cities_series.index
                     print(f"Filtering for top {n_cities} cities: {list(top_cities)}")
                     filtered_df = filtered_df[filtered_df['city_name'].isin(top_cities)]

                 if filtered_df.empty:
                     print("Warning: City filtering resulted in an empty DataFrame. Skipping heatmap.")
                     print("-" * 20)
                     return
            else:
                 print("Warning: No data after date filtering, cannot determine top cities.")

        else:
             print("Warning: 'city_name' column not found for city filtering.")
    elif n_cities is not None and n_cities <= 0:
         print("Warning: Invalid value for n_cities. Including all cities within the date range.")


    # Ensure filtered_df is not empty before creating pivot
    if filtered_df.empty:
        print("Warning: Filtered DataFrame is empty before creating pivot table. Skipping heatmap.")
        print("-" * 20)
        return

    # Pivot the data: rows = cities, columns = dates, values = AQI
    # Aggregation (mean) is used in case of multiple entries per city/date
    # Use only the date part for the column labels to keep it cleaner
    if filtered_df['datetime'].dt.date.nunique() > 100 or filtered_df['city_name'].nunique() > 100:
         print("Note: The filtered heatmap still contains many unique dates or cities. It might be large.")

    # Drop rows with NaN in essential columns before pivot
    pivot_data = filtered_df.dropna(subset=['city_name', 'datetime', 'main.aqi']).copy()

    if pivot_data.empty:
        print("Warning: No valid data after dropping NaNs for pivot table. Skipping heatmap.")
        print("-" * 20)
        return

    pivot_table = pivot_data.pivot_table(index='city_name',
                                           columns=pivot_data['datetime'].dt.date, # Use date part for column names
                                           values='main.aqi',
                                           aggfunc='mean')

    if pivot_table.empty:
         print("Warning: Pivot table is empty after creation. Skipping heatmap.")
         print("-" * 20)
         return

    # Sort by city name (optional)
    pivot_table = pivot_table.sort_index()

    # Adjust figure size dynamically based on the filtered data dimensions (heuristic)
    fig_width = max(pivot_table.shape[1] * 0.3, 10)
    fig_height = max(pivot_table.shape[0] * 0.4, 6)
    fig_width = min(fig_width, 40)
    fig_height = min(fig_height, 30)

    # Create the heatmap
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap='RdYlGn_r', linewidths=0.1, linecolor='gray',
                cbar_kws={'label': 'Average AQI (1=Good → 5=Hazardous)'})
    plt.title('Average AQI Heatmap (City vs Date)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('City')
    plt.tight_layout()
    plt.show()
    print("-" * 20)


def plot_weekly_aqi_trend(df):
    """Plots the weekly average AQI trend over time."""
    if 'datetime' not in df.columns or 'main.aqi' not in df.columns:
        print("Error: Required columns ('datetime', 'main.aqi') not found for weekly AQI trend plot.")
        print("-" * 20)
        return

    print("\n--- Weekly Average AQI Trend Over Time ---")

    # Create a temporary DataFrame with essential columns and drop NaNs
    temp_df = df[['datetime', 'main.aqi']].dropna().copy()

    if temp_df.empty:
        print("Warning: No valid data after dropping NaNs for weekly AQI trend plot.")
        print("-" * 20)
        return

    # Ensure datetime is processed and set as index for resampling
    if not pd.api.types.is_datetime64_any_dtype(temp_df['datetime']):
         temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce')
         temp_df.dropna(subset=['datetime'], inplace=True) # Drop NaTs after coercion

    if temp_df.empty:
         print("Warning: No valid datetime data after processing for weekly AQI trend plot.")
         print("-" * 20)
         return

    temp_df = temp_df.set_index('datetime')

    # Resample to weekly frequency ('W') and calculate the mean
    weekly_aqi = temp_df['main.aqi'].resample('W').mean().reset_index()

    if weekly_aqi.empty:
         print("Warning: Resampling resulted in an empty DataFrame for weekly AQI trend plot.")
         print("-" * 20)
         return

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=weekly_aqi, x='datetime', y='main.aqi', marker='o', linestyle='-', markersize=4)
    plt.title('Weekly Average AQI Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average AQI')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("-" * 20)

def detect_aqi_outliers(df):
    """
    Detects outliers in the 'main.aqi' column using the IQR method
    and prints summary information.
    """
    print("\n--- AQI Outlier Detection (IQR Method) ---")

    if 'main.aqi' not in df.columns:
        print("Error: 'main.aqi' column not found for outlier detection.")
        print("-" * 20)
        return

    # Work with non-null AQI values for quantile calculation
    aqi_data = df['main.aqi'].dropna()

    if aqi_data.empty:
        print("Warning: 'main.aqi' column is empty or contains only missing values. Cannot detect outliers.")
        print("-" * 20)
        return

    # Calculate Q1, Q3, and IQR
    Q1 = aqi_data.quantile(0.25)
    Q3 = aqi_data.quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds (1.5 * IQR rule)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Q1 (25th percentile): {Q1:.2f}")
    print(f"Q3 (75th percentile): {Q3:.2f}")
    print(f"IQR (Interquartile Range): {IQR:.2f}")
    print(f"Lower bound for outliers: {lower_bound:.2f}")
    print(f"Upper bound for outliers: {upper_bound:.2f}")

    # Identify outliers
    outliers = df[(df['main.aqi'] < lower_bound) | (df['main.aqi'] > upper_bound)]

    total_records = len(df.dropna(subset=['main.aqi'])) # Count records with valid AQI
    num_outliers = len(outliers)

    print(f"\nTotal records with valid AQI: {total_records}")
    print(f"Number of detected outliers: {num_outliers}")

    if total_records > 0:
        percentage_outliers = (num_outliers / total_records) * 100
        print(f"Percentage of outliers: {percentage_outliers:.2f}%")

    if num_outliers > 0:
        print("\nExamples of detected outliers:")
        # Display the first few outlier rows with relevant columns
        print(outliers[['datetime', 'city_name', 'main.aqi']].head())
    else:
        print("\nNo outliers detected based on the IQR method.")

    print("-" * 20)


if __name__ == "__main__":
    # Set Seaborn theme
    sns.set(style="whitegrid")

    # --- Main Analysis Flow ---
    data_filepath = "final_dataset.csv"
    df = load_data(data_filepath)

    if df is not None:
        # Preprocess data - crucial for date handling
        df = preprocess_data(df)

        # Check if 'datetime' column is valid and not empty after preprocessing
        if 'datetime' in df.columns and not df['datetime'].empty and df['datetime'].notna().any():

            display_basic_info(df) # Display info after preprocessing

            # Perform visualizations and analysis
            plot_aqi_distribution(df)
            plot_records_per_year(df)
            plot_aqi_trend_top_cities(df)
            plot_aqi_boxplot_top_cities(df)
            plot_monthly_aqi_heatmap_top_cities(df)

            # --- Plot Weekly Average AQI Trend ---
            plot_weekly_aqi_trend(df)

            # --- Detect AQI Outliers ---
            detect_aqi_outliers(df)

            # --- Limit the City vs Date Heatmap to the last 30 days and top 10 cities ---

            # Find the most recent date in the dataset from non-NaT values
            most_recent_date_full = df['datetime'].max()

            # Ensure most_recent_date is not NaT
            if pd.isna(most_recent_date_full):
                 print("\nCannot plot City vs Date Heatmap for the last 30 days: No valid datetime values found in the dataset.")
            else:
                most_recent_date = most_recent_date_full.date()
                # Calculate the start date for the last 30 days
                # Subtracting 29 days includes the most recent date, making it a 30-day range
                start_date_last_30_days = most_recent_date - pd.Timedelta(days=29)

                print(f"\nPreparing City vs Date Heatmap for the last 30 days ({start_date_last_30_days} to {most_recent_date}) and top 10 cities.")

                # Call the heatmap function with the calculated dates and n_cities
                plot_city_date_heatmap(
                    df,
                    start_date=start_date_last_30_days,
                    end_date=most_recent_date,
                    n_cities=10
                )
        else:
            print("\nSkipping datetime-based plots and outlier detection: 'datetime' column is missing or contains no valid data after preprocessing.")
            # If datetime is not available, still attempt non-datetime specific EDA if needed
            # Here we assume datetime is essential for most subsequent steps.


    else:
        print("Data loading failed. Exiting.")