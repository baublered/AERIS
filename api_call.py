## This script tests the OpenWeatherMap API key by fetching weather data for a specified city.
import requests
import pandas as pd
import os

def get_aqi_from_csv_coords(api_key, city='Antipolo'):
    # Construct the path to the CSV file
    csv_path = 'city_coordinates_from_dataset.csv'

    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found in the current directory.")
        return

    # Load coordinates from the CSV file
    try:
        city_coords_df = pd.read_csv(csv_path)
        # Find the row for the specified city (case-insensitive)
        city_data = city_coords_df[city_coords_df['city_name'].str.lower() == city.lower()]

        if city_data.empty:
            print(f"Error: City '{city}' not found in '{csv_path}'.")
            return

        # Extract latitude and longitude
        lat = city_data.iloc[0]['latitude']
        lon = city_data.iloc[0]['longitude']
        
        print(f"Using coordinates for {city} from CSV: lat={lat}, lon={lon}")

    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # Get air quality data using coordinates from the CSV
    aqi_url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}'
    aqi_response = requests.get(aqi_url)

    if aqi_response.status_code == 200:
        aqi_data = aqi_response.json()
        aqi = aqi_data['list'][0]['main']['aqi']
        pm25 = aqi_data['list'][0]['components']['pm2_5']  # PM2.5 value
        print(f"Main AQI: {aqi}")
        print(f'PM2.5 current level: ', pm25)
    else:
        print(f"Error fetching AQI data. Status code: {aqi_response.status_code}")
        print(f"Response: {aqi_response.text}")
        

if __name__ == '__main__':
    api_key = '046db00705c9a85bd3daa1f61ea04d4a'  # Replace with your actual API key
    get_aqi_from_csv_coords(api_key, city='Antipolo') # You can change the city here
#NOTE: this script is for testing purposes only. It fetches weather and AQI data for a specified city using the OpenWeatherMap API.
# IT WORKS TYLLLLLLL!!! :D YIPPEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE