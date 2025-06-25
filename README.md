# 🌏 AERIS — Dynamic Air Quality Forecasting for the Philippines

**AERIS** is a desktop application that provides real-time air quality forecasts for 138 major cities in the Philippines. It uses a sophisticated two-step machine learning pipeline to predict PM2.5 concentrations and overall Air Quality Index (AQI) levels for the next three days, presenting the data in an intuitive interface.

---

## 📌 Features

-   **Dynamic Two-Step Forecasting**: Predicts future PM2.5 values first, then uses those predictions to forecast a more accurate and responsive AQI.
-   **Multi-Day Forecasts**: Generates recursive forecasts for the next 3 days.
-   **Dual Model System**: Utilizes two separate LightGBM models for PM2.5 and AQI prediction.
-   **Comprehensive Visualization**: The UI displays forecasts for both AQI and PM2.5 in separate graphs, including real-time values for comparison.
-   **Forecast Drift Analysis**: Calculates and displays metrics (MAE, RMSE) showing how much the forecast deviates from the current real-time value.
-   **Broad City Coverage**: Supports forecasting for 138 major Philippine cities.
-   **Intuitive Desktop UI**: Built with Python's native Tkinter library for a simple and responsive user experience.

---

## 📂 Project Structure

```
AERIS/
├── aeris_forecast.py           # Main application file with Tkinter GUI and forecasting logic
├── train_aqi_model.py          # Script to train the final AQI forecasting model
├── train_pm25_model.py         # Script to train the intermediate PM2.5 forecasting model
├── eda_aeris.py                # Script for Exploratory Data Analysis of the dataset
│
├── aqi_forecasting_model.pkl   # Trained LightGBM model for AQI
├── used_features.pkl           # Features used by the AQI model
├── pm25_forecasting_model.pkl  # Trained LightGBM model for PM2.5
├── pm25_used_features.pkl      # Features used by the PM2.5 model
│
├── final_dataset.csv           # Consolidated historical dataset for training
├── city_coordinates_from_dataset.csv # City-to-coordinate mapping
├── city_label_encoder.pkl      # Saved encoder for city names
│
└── eda_plots/                  # Directory for output from the EDA script
```

---

## ⚙️ How It Works

AERIS employs a two-stage recursive forecasting process to generate its predictions:

1.  **Real-time Data Fetch**: The application retrieves the current air pollution data (including AQI and all pollutant components) for a user-selected city from the OpenWeatherMap API.
2.  **PM2.5 Forecast (Stage 1)**: The `pm25_forecasting_model` predicts the PM2.5 value for Day 1. This predicted value is then used as a feature (a "lag" value) to predict the PM2.5 for Day 2, and so on, for three days.
3.  **AQI Forecast (Stage 2)**: The `aqi_forecasting_model` takes the *predicted* PM2.5 values from Stage 1, along with other real-time data, to forecast the final AQI for each of the next three days. This makes the AQI forecast highly sensitive to changes in predicted pollution.
4.  **Display Results**: The application displays the real-time data and the 3-day forecasts for both PM2.5 and AQI in the UI, complete with graphs and forecast drift metrics.

---

## 📊 Model Overview

The system uses two distinct LightGBM Regressor models:

1.  **PM2.5 Forecasting Model**:
    -   **Objective**: Predict the `components.pm2_5` value for a future day.
    -   **Key Inputs**: City coordinates, date features (month, day of week), and lag features of past PM2.5 values.
2.  **AQI Forecasting Model**:
    -   **Objective**: Predict the final `main.aqi` value (1-5).
    -   **Key Inputs**: City coordinates, date features, all real-time pollutant values, and the **predicted PM2.5 value** from the first model. This model was intentionally trained *without* AQI lag features to force it to be more dynamic and responsive to pollutant changes.

---

## 🧪 Requirements

Key libraries used in this project:

-   `lightgbm`
-   `pandas`
-   `numpy`
-   `requests`
-   `scikit-learn`
-   `joblib`
-   `matplotlib`
-   `tkinter` (built-in with Python)

### Installation

1.  **Clone the repository**
    ```bash
    git clone <repository_url>
    cd airforsee
    ```
2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Get an OpenWeatherMap API Key**
    -   Sign up for a free account on the [OpenWeatherMap website](https://openweathermap.org/).
    -   Generate an API key from your account dashboard.
    -   Ensure you have access to their **Air Pollution API**.
4.  **Set up the API Key**
    -   Open `aeris_forecast.py` and replace the placeholder `"YOUR_API_KEY_HERE"` with your actual OpenWeatherMap API key.

---

## ▶️ Usage

1.  **Run the Application**
    ```bash
    python aeris_forecast.py
    ```
2.  The Tkinter GUI will open.
3.  Select a major Philippine city from the dropdown menu.
4.  Click the **"Get Forecast"** button.
5.  The application will display the 3-day forecast details, graphs for AQI and PM2.5, and a chart showing the forecast drift from current values.

---

## 🙏 Contributing

Contributions are welcome! If you'd like to contribute to AERIS, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes and commit them (`git commit -m 'Add your feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Create a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.