import pandas as pd
import yfinance as yf
from prophet import Prophet
import logging
import streamlit as st
from datetime import datetime

# Configure Streamlit and Logging
st.set_page_config(page_title="Test App with Download", layout="wide")

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Test Script
def main():
    st.title("🔍 Minimal Testing Script with Immediate Download")
    
    # Sidebar for inputs
    symbol = st.sidebar.text_input("Enter stock symbol", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    run_button = st.sidebar.button("Fetch & Download Data")
    
    if run_button:
        st.write(f"Fetching stock data for: {symbol}")
        st.write(f"Start Date: {start_date}, End Date: {end_date}")
        
        # Step 1: Fetch Raw Data
        st.subheader("Step 1: Fetch Raw Data")
        data = fetch_stock_data(symbol, start_date, end_date)
        if data is None:
            st.error("Data fetching failed. Check logs.")
            return
        st.write(data.head())
        st.success("Data fetching passed.")
        
        # Provide download link for raw data
        st.download_button(
            label="Download Raw Data",
            data=data.to_csv(index=False),
            file_name=f"{symbol}_raw_data.csv",
            mime="text/csv"
        )
        
        # Step 2: Preprocess Data
        st.subheader("Step 2: Preprocess Data")
        data = preprocess_data(data)
        if data is None or data.empty:
            st.error("Data preprocessing failed. Check logs.")
            return
        st.write(data.head())
        st.success("Data preprocessing passed.")
        
        # Provide download link for preprocessed data
        st.download_button(
            label="Download Preprocessed Data",
            data=data.to_csv(index=False),
            file_name=f"{symbol}_preprocessed_data.csv",
            mime="text/csv"
        )

def fetch_stock_data(symbol, start_date, end_date):
    logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            logging.error("No data fetched.")
            return None
        return data.reset_index()
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

if __name__ == "__main__":
    main()
