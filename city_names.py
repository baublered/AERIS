# Extract and save Philippine city names to a text file
import pandas as pd

df = pd.read_csv("cleaned_air_quality_data.csv")
cities = sorted(df["city_name"].unique())

with open("ph_city_list.txt", "w") as f:
    for city in cities:
        f.write(city + "\n")

print(f"✅ Saved {len(cities)} cities to ph_city_list.txt")