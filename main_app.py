#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:19:58 2024

@author: thodoreskourtales
"""

import streamlit as st

def main():
    st.title("📊 Stock Forecasting Application")
    st.markdown("""
    Welcome to the **Stock Forecasting Application**!
    
    This app allows you to:
    - Fetch raw stock data from Yahoo Finance.
    - Clean and preprocess the data.
    - Train a forecasting model using Facebook Prophet.
    - Generate future forecasts with interactive visualizations.
    
    **Navigate to the Forecasting Script to begin the process.**
    """)
    
    # Navigation button
    if st.button("Go to Forecasting Script"):
         st.query_params(page="Forecasting_Script")

if __name__ == "__main__":
    main()
