#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App for Stock Forecasting

Created on Fri Dec 20 22:17:33 2024

@author: thodoreskourtales
"""

import subprocess
import sys
import importlib
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

from prophet import Prophet
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fftpack import fft

# Ensure required packages are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "numpy", "pandas", "scipy", "yfinance", "matplotlib",
    "scikit-learn", "prophet", "seaborn",
    "keras", "plotly", "statsmodels", "streamlit"
]
for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        install(package)

# Configure Streamlit
st.set_page_config(page_title="Stock Forecasting App", layout="wide")

# Main Streamlit App
def main():
    st.title("📈 Stock Forecasting App")
    st.markdown("""
        This application allows you to:
        - **Fetch** historical stock data.
        - **Clean** and **download** raw and processed data.
        - **Forecast** future stock prices using Prophet.
    """)

    # Sidebar for user inputs
    st.sidebar.header("User Inputs")
    symbol = st.sidebar.text_input("Enter stock symbol", "MSFT").upper()
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    forecast_days = st.sidebar.number_input("Number of days to forecast", min_value=1, max_value=365, value=30)
    run_button = st.sidebar.button("Run Forecast")

    if run_button:
        st.header(f"🔍 Processing {symbol}...")

        # Step 1: Fetch Raw Data
        st.subheader("Step 1: Fetch Raw Data")
        raw_data = fetch_stock_data(symbol, start_date, end_date)
        if raw_data is None:
            st.error(f"Failed to fetch data for {symbol}. Please check the symbol and date range.")
            return
        st.write("**Raw Data Preview:**")
        st.write(raw_data.head())

        # Download Raw Data
        st.download_button(
            label="Download Raw Data",
            data=raw_data.to_csv(index=False).encode('utf-8'),
            file_name=f"{symbol}_raw_data.csv",
            mime="text/csv"
        )

        # Step 2: Clean Data
        st.subheader("Step 2: Clean Data")
        cleaned_data = clean_data(raw_data)
        if cleaned_data is None or cleaned_data.empty:
            st.error("Data cleaning failed. Please check the logs for more details.")
            return
        st.write("**Cleaned Data Preview:**")
        st.write(cleaned_data.head())

        # Download Cleaned Data
        st.download_button(
            label="Download Cleaned Data",
            data=cleaned_data.to_csv(index=False).encode('utf-8'),
            file_name=f"{symbol}_cleaned_data.csv",
            mime="text/csv"
        )

        # Step 3: Train Prophet Model
        st.subheader("Step 3: Train Prophet Model")
        model, holidays = train_prophet_model(cleaned_data)
        if model is None:
            st.error("Failed to train Prophet model.")
            return
        st.success("Prophet model trained successfully.")

        # Step 4: Forecasting
        st.subheader(f"Step 4: Forecasting for Next {forecast_days} Days")
        forecast = forecast_prices(model, holidays, forecast_days)
        if forecast is None:
            st.error("Failed to generate forecast.")
            return
        st.write("**Forecast Data Preview:**")
        st.write(forecast.tail())

        # Plot Forecast
        st.subheader("📊 Forecast Plot")
        plot_forecast_streamlit(cleaned_data, forecast, symbol)

# Function to fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No data found for {symbol} between {start_date} and {end_date}.")
            return None
        return data.reset_index()
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to clean data
def clean_data(data):
    try:
        st.write("**Raw Data Column Names:**")
        st.write(data.columns.tolist())

        # Step 1: Handle multi-level headers if present
        if isinstance(data.columns, pd.MultiIndex):
            st.warning("Detected multi-level header. Flattening...")
            data.columns = ['_'.join(filter(None, map(str, col))) for col in data.columns]
            st.write("**Flattened Column Names:**")
            st.write(data.columns.tolist())

        # Step 2: Dynamically identify 'Date' and 'Close' columns
        date_column = next((col for col in data.columns if "Date" in col or "date" in col), None)
        close_column = next((col for col in data.columns if "Close" in col or "close" in col), None)

        if not date_column or not close_column:
            st.error(f"Unable to identify required columns. Available columns: {data.columns.tolist()}")
            return None

        # Step 3: Rename columns to 'ds' and 'y' for Prophet
        data = data.rename(columns={date_column: "ds", close_column: "y"})
        st.write("**Renamed Columns:**")
        st.write(data.columns.tolist())

        # Step 4: Convert 'ds' to datetime and 'y' to numeric
        data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
        data['y'] = pd.to_numeric(data['y'], errors='coerce')

        # Step 5: Drop rows with NaN in 'ds' or 'y'
        data = data.dropna(subset=['ds', 'y'])
        st.write("**Data After Cleaning:**")
        st.write(data.head())

        # Final validation
        st.write("**Cleaned Data Types:**")
        st.write(data.dtypes)

        return data

    except Exception as e:
        st.error(f"Error during data cleaning: {e}")
        return None

# Function to add holiday effects
def add_holiday_effects(model, data):
    holidays = pd.DataFrame({
        'holiday': 'major_holidays',
        'ds': pd.to_datetime(['2023-01-01', '2023-07-04', '2023-12-25']),
        'lower_window': 0,
        'upper_window': 1,
    })
    model.add_country_holidays(country_name='US')
    data['holiday_effects'] = data['ds'].apply(lambda x: 1 if x in holidays['ds'].values else 0)
    return model, data, holidays

# Function to train Prophet model
def train_prophet_model(data):
    try:
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model, data, holidays = add_holiday_effects(model, data)
        model.fit(data)
        return model, holidays
    except Exception as e:
        st.error(f"Error training Prophet model: {e}")
        return None, None

# Function to forecast prices
def forecast_prices(model, holidays, periods):
    try:
        future = model.make_future_dataframe(periods=periods)
        future['holiday_effects'] = future['ds'].apply(lambda x: 1 if x in holidays['ds'].values else 0)
        forecast = model.predict(future)

        # Ensure forecast values are not negative
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
        forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(x, 0))
        forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: max(x, 0))

        return forecast
    except Exception as e:
        st.error(f"Error during price forecasting: {e}")
        return None

# Function to plot forecast in Streamlit
def plot_forecast_streamlit(data, forecast, symbol, actual_data=None):
    try:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot historical data
        ax.plot(data['ds'], data['y'], label='Historical Stock Price', color='blue')

        # Plot forecast data
        forecast_positive = forecast[forecast['yhat'] > 0]
        ax.plot(forecast_positive['ds'], forecast_positive['yhat'], label='Forecasted Price', color='red')
        ax.fill_between(forecast_positive['ds'], forecast_positive['yhat_lower'], forecast_positive['yhat_upper'], color='pink', alpha=0.5)

        # Plot actual future data if available
        if actual_data is not None:
            ax.plot(actual_data['ds'], actual_data['y'], label='Actual Future Price', color='green')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Forecast for {symbol}')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error plotting forecast for {symbol}: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
