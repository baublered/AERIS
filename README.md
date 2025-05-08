# 🌏 AERIS — Air Quality Forecasting in the Philippines

**AERIS** is a real-time air quality forecasting system that predicts AQI levels and classifies air quality conditions for 138 major cities in the Philippines. It combines real-time environmental data with a machine learning model to generate forecasts for the next 3 days — similar to a weather app.

---

## 📌 Features

- 🔮 Real-time AQI forecasting based on OpenWeatherMap API.
- 📅 Multi-day (recursive) forecasts up to 3 days ahead.
- 🧠 LightGBM model trained on historical weather and pollution data.
- 🧪 AQI classification: Good, Fair, Moderate, Poor, or Very Poor.
- 🗺️ Forecast for 138 major Philippine cities.
- 🖥️ Simple and responsive desktop UI built with Tkinter.

---

## 📂 Project Structure
AERIS/
├── aeris_forecast.py # Core forecasting logic
├── main.py # Tkinter-based user interface
├── aqi_forecasting_model.pkl # Trained LightGBM model
├── used_features.pkl # Features used during model training
├── cleaned_air_quality_data.csv # Historical dataset (Nov 2023 – April 2025)
├── city_coordinates_from_dataset.csv # City-to-coordinate mapping


---

## ⚙️ How It Works

1. **User Input**: A city name is entered via the interface.
2. **Real-time Data**: AQI and weather data are retrieved from the **OpenWeatherMap API**.
3. **Feature Engineering**: Key features are constructed (e.g. city label, season, weather metrics).
4. **Forecasting**: AQI levels are predicted recursively for 3 future days.
5. **AQI Classification**:
   - **Good** (1)
   - **Fair** (2)
   - **Moderate** (3)
   - **Poor** (4)
   - **Very Poor** (5)
6. **Display**: Predictions and AQI labels are shown in the UI.

---

## 📊 Model Overview

- **Model**: LightGBM Regressor
- **Training Period**: November 2023 – April 2025
- **Targets**: AQI Level
- **Inputs**: Weather conditions, city, date, and engineered features
- **Output**: 3-day forecast of AQI values with AQI classification

---

## 🧪 Requirements
Main libraries:

- lightgbm
- pandas
- numpy
- requests
- scikit-learn
- tkinter (built-in)

Installation
1. Clone the repository
    git clone <repository_url>
    cd AERIS
2. Install dependencies
    pip install -r requirements.txt
3. Get an OpenWeatherMap API Key
    Sign up for a free account on the OpenWeatherMap website.
    Generate an API key from your account dashboard
    Make sure you have access to their Air Pollution API
4. Set up the API Key
    You will need to provide your OpenWeatherMap API key to the application.
    Replace your own API key in aeris_forecast.py

▶️ Usage
1. Run the Application
    python aeris_forecast.py
2. The Tkinter GUI will open.
3. Enter the name of a major Philippine city in the input field.
4. Click the "Get Forecast" button.
5. The application will display the 3-day AQI forecast and AQI classification for the selected city.

🙏 Contributing
Contributions are welcome! If you'd like to contribute to AERIS, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).   
Make your changes and commit them (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).   
Create a Pull Request.

📄 License
This project is licensed under the MIT License.