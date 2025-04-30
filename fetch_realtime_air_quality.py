import requests
import pandas as pd
import time

API_KEY = "62067aa28e490926060739d6420d490a7ab08c2f"
INPUT_FILE = "city_coordinates_from_dataset.csv"
OUTPUT_FILE = "real_time_air_quality.csv"

def get_real_time_data(lat, lon):
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["status"] == "ok":
            aqi = data["data"].get("aqi", "N/A")
            iaqi = data["data"].get("iaqi", {})
            pm25 = iaqi.get("pm25", {}).get("v", "N/A")
            pm10 = iaqi.get("pm10", {}).get("v", "N/A")
            return aqi, pm25, pm10
        else:
            return "N/A", "N/A", "N/A"
    except Exception as e:
        print(f"Error for lat={lat}, lon={lon}: {e}")
        return "N/A", "N/A", "N/A"

def main():
    df = pd.read_csv(INPUT_FILE)
    results = []

    for index, row in df.iterrows():
        city = row["city_name"]
        lat = row["latitude"]
        lon = row["longitude"]
        print(f"🌍 Fetching real-time AQI for {city} ({lat}, {lon})")
        aqi, pm25, pm10 = get_real_time_data(lat, lon)
        results.append({
            "city_name": city,
            "latitude": lat,
            "longitude": lon,
            "aqi": aqi,
            "pm25": pm25,
            "pm10": pm10
        })
        time.sleep(1)  # To respect WAQI API rate limits

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Real-time air quality data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
