import pandas as pd

INPUT_FILE = "cleaned_air_quality_data.csv"
OUTPUT_FILE = "city_coordinates_from_dataset.csv"

def main():
    # Load the dataset
    df = pd.read_csv(INPUT_FILE)

    # Extract only relevant columns and drop duplicates
    unique_coords = df[["city_name", "coord.lat", "coord.lon"]].drop_duplicates()

    # Rename columns for clarity (optional)
    unique_coords = unique_coords.rename(columns={
        "coord.lat": "latitude",
        "coord.lon": "longitude"
    })

    # Drop rows with missing or NaN values
    unique_coords = unique_coords.dropna(subset=["latitude", "longitude"])

    # Save to new CSV
    unique_coords.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Saved unique city coordinates to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
