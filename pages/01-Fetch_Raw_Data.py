#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:20:28 2024

@author: thodoreskourtales
"""

# pages/1_Fetch_Raw_Data.py

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

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

def main():
    st.header("🔍 Step 1: Fetch Raw Data")
    
    # User Inputs
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOG)", "MSFT").upper()
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.today())
    fetch_button = st.button("Fetch Data")
    
    if fetch_button:
        with st.spinner(f"Fetching data for {symbol}..."):
            data = fetch_stock_data(symbol, start_date, end_date)
            if data is not None:
                st.success("Data fetched successfully!")
                st.write("**Raw Data Preview:**")
                st.dataframe(data.head())
                
                # Display column names
                st.write("**Raw Data Column Names:**")
                st.write(data.columns.tolist())
                
                # Handle multi-level headers if present
                if isinstance(data.columns, pd.MultiIndex):
                    st.warning("Detected multi-level header. Flattening...")
                    data.columns = ['_'.join(filter(None, map(str, col))) for col in data.columns]
                    st.write("**Flattened Column Names:**")
                    st.write(data.columns.tolist())
                
                # Rename columns for Prophet
                date_column = 'Date' if 'Date' in data.columns else None
                # Assuming 'Close' column is present; adjust if different
                close_column_candidates = [col for col in data.columns if 'Close' in col]
                close_column = close_column_candidates[0] if close_column_candidates else None
                
                if date_column and close_column:
                    data = data.rename(columns={date_column: "ds", close_column: "y"})
                    st.write("**Renamed Columns:**")
                    st.write(data.columns.tolist())
                    
                    # Save to session_state
                    st.session_state['raw_data'] = data
                    st.session_state['symbol'] = symbol  # Store the symbol
                else:
                    st.error("Unable to identify required columns 'Date' and 'Close'.")
            else:
                st.error("Failed to fetch data. Please check the stock symbol and date range.")
    st.markdown("---")
    st.markdown("### Select a Step:")
    
    # Navigation Buttons with Correct Page Titles
    if st.button("Step 1: Fetch Raw Data"):
        switch_page("fetch raw data")
    
    if st.button("Step 2: Clean Data"):
        switch_page("clean data")
    
    if st.button("Step 3: Train Prophet Model"):
        switch_page("train prophet")
    
    if st.button("Step 4: Forecast and Anomaly Detection"):
        switch_page("forecast anomaly detection")

if __name__ == "__main__":
    main()
