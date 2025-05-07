import requests

api_key = "046db00705c9a85bd3daa1f61ea04d4a"
latitude = 14.62
longitude = 121.08
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={latitude}&lon={longitude}&appid={api_key}"

response = requests.get(url)
data = response.json()
print(data)