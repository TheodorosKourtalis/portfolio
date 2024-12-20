import pandas as pd
import yfinance as yf
from prophet import Prophet
import logging
import streamlit as st
from datetime import datetime, timedelta

# Configure Streamlit and Logging
st.set_page_config(page_title="Test App", layout="wide")

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Test Script
def main():
    st.title("🔍 Minimal Testing Script")
    
    # Sidebar for inputs
    symbol = st.sidebar.text_input("Enter stock symbol", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    periods = st.sidebar.number_input("Days to Forecast", min_value=1, max_value=365, value=30)
    run_button = st.sidebar.button("Run Tests")
    
    if run_button:
        st.write(f"Testing with stock: {symbol}")
        st.write(f"Start Date: {start_date}, End Date: {end_date}")
        
        # Step 1: Test Data Fetching
        st.subheader("Step 1: Fetch Data")
        data = fetch_stock_data(symbol, start_date, end_date)
        if data is None:
            st.error("Data fetching failed. Check logs.")
            return
        st.write(data.head())
        st.success("Data fetching passed.")
        
        # Step 2: Test Preprocessing
        st.subheader("Step 2: Preprocess Data")
        data = preprocess_data(data)
        if data is None or data.empty:
            st.error("Data preprocessing failed. Check logs.")
            return
        st.write(data.head())
        st.success("Data preprocessing passed.")
        
        # Step 3: Test Prophet Model
        st.subheader("Step 3: Train Prophet Model")
        model, holidays = train_prophet_model(data)
        if model is None:
            st.error("Prophet model training failed. Check logs.")
            return
        st.success("Prophet model training passed.")
        
        # Step 4: Test Forecasting
        st.subheader("Step 4: Generate Forecast")
        forecast = forecast_prices(model, holidays, periods)
        if forecast is None:
            st.error("Forecasting failed. Check logs.")
            return
        st.write(forecast.head())
        st.success("Forecasting passed.")

def fetch_stock_data(symbol, start_date, end_date):
    logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            logging.error("No data fetched.")
            return None
        return data[['Close']].reset_index()
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def preprocess_data(data):
    logging.info("Preprocessing data...")
    try:
        if 'Date' not in data.columns:
            logging.error("Missing 'Date' column.")
            return None
        if 'Close' not in data.columns:
            logging.error("Missing 'Close' column.")
            return None
        
        data = data.rename(columns={'Date': 'ds', 'Close': 'y'})
        data['y'] = data['y'].replace(0, pd.NA).ffill()
        data = data.dropna(subset=['y', 'ds'])
        data['y'] = pd.to_numeric(data['y'], errors='coerce')
        return data.dropna()
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        return None

def train_prophet_model(data):
    logging.info("Training Prophet model...")
    try:
        model = Prophet(daily_seasonality=True)
        model.add_country_holidays(country_name='US')
        model.fit(data)
        return model, None
    except Exception as e:
        logging.error(f"Error training Prophet model: {e}")
        return None, None

def forecast_prices(model, holidays, periods):
    logging.info(f"Generating forecast for {periods} days...")
    try:
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast
    except Exception as e:
        logging.error(f"Error generating forecast: {e}")
        return None

if __name__ == "__main__":
    main()
