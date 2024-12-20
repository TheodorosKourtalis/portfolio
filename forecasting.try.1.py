import pandas as pd
import yfinance as yf
from prophet import Prophet
import streamlit as st
from datetime import datetime

# Configure Streamlit
st.set_page_config(page_title="Data Handling Test", layout="wide")

# Main Function
def main():
    st.title("🔍 Handle Data Structure & Download Processed File")
    
    # Sidebar for inputs
    symbol = st.sidebar.text_input("Enter stock symbol", "MSFT")
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    run_button = st.sidebar.button("Fetch & Process Data")
    
    if run_button:
        st.write(f"Fetching stock data for: {symbol}")
        st.write(f"Start Date: {start_date}, End Date: {end_date}")
        
        # Step 1: Fetch Raw Data
        st.subheader("Step 1: Fetch Raw Data")
        data = fetch_stock_data(symbol, start_date, end_date)
        if data is None:
            st.error("Data fetching failed. Check logs.")
            return
        st.write("Raw Data (before cleaning):", data.head())
        
        # Provide download link for raw data
        st.download_button(
            label="Download Raw Data",
            data=data.to_csv(index=False),
            file_name=f"{symbol}_raw_data.csv",
            mime="text/csv"
        )
        
        # Step 2: Clean Data
        st.subheader("Step 2: Clean Data")
        data = clean_data(data)
        if data is None or data.empty:
            st.error("Data cleaning failed. Check logs.")
            return
        st.write("Cleaned Data:", data.head())
        
        # Provide download link for cleaned data
        st.download_button(
            label="Download Cleaned Data",
            data=data.to_csv(index=False),
            file_name=f"{symbol}_cleaned_data.csv",
            mime="text/csv"
        )

def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.error("No data fetched.")
            return None
        return data.reset_index()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def clean_data(data):
    try:
        # Display raw data
        st.write("Raw data preview:")
        st.write(data.head())
        st.write("Raw data column names:", list(data.columns))

        # Step 1: Adjust for misaligned headers
        if not pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
            st.warning("Detected possible extra header row. Adjusting headers...")
            data.columns = data.iloc[0]  # Set second row as header
            data = data[1:]  # Drop the first row
            st.write("Adjusted headers preview:")
            st.write(data.head())

        # Step 2: Reset index
        data = data.reset_index(drop=True)

        # Step 3: Rename columns for standardization
        if "Date" in data.columns and "Close" in data.columns:
            data = data.rename(columns={"Date": "ds", "Close": "y"})
        else:
            st.error("Missing required columns 'Date' or 'Close'. Please check your data format.")
            return None

        # Step 4: Validate column types
        st.write("Validating column types...")
        st.write("Column types before conversion:", data.dtypes)

        # Convert 'ds' to datetime and 'y' to numeric
        data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
        data['y'] = pd.to_numeric(data['y'], errors='coerce')

        # Step 5: Drop rows with NaN in critical columns
        if 'ds' not in data or 'y' not in data:
            st.error("Critical columns are missing after processing. Exiting.")
            return None

        data = data.dropna(subset=['ds', 'y'])
        st.write("Data after cleaning:")
        st.write(data.head())

        # Step 6: Debugging final data
        st.write("Final cleaned data types:")
        st.write(data.dtypes)

        return data

    except Exception as e:
        st.error(f"Error during data cleaning: {e}")
        st.write("Detailed error context:")
        st.write(e)
        return None
if __name__ == "__main__":
    main()
