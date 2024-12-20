#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 23:07:02 2024

@author: thodoreskourtales
"""

import streamlit as st
from streamlit_extras.switch_page_button import switch_page

def main():
    st.title("📈 Forecasting Workflow")
    st.markdown("""
    Welcome to the **Forecasting Workflow**! Follow the steps below to generate your stock price forecast:
    
    1. **Fetch Raw Data:** Retrieve stock data from Yahoo Finance.
    2. **Clean Data:** Preprocess the data to prepare it for modeling.
    3. **Train Prophet Model:** Train a forecasting model using Facebook Prophet.
    4. **Forecast:** Generate future forecasts using FB Prophet and visualize them interactively.
    """)
    
    st.markdown("---")
    st.markdown("### Select a Step to Proceed:")
    
    # Navigation Buttons with Correct Page Titles
    if st.button("Step 1: Fetch Raw Data"):
        switch_page("fetch raw data")
    
    if st.button("Step 2: Clean Data"):
        switch_page("clean data")
    
    if st.button("Step 3: Train Prophet Model"):
        switch_page("train prophet")
    
    if st.button("Step 4: Forecast"):
        switch_page("forecast")

if __name__ == "__main__":
    main()
