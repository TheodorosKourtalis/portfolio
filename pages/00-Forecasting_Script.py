#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 23:07:02 2024

Author: Theodoros Kourtalis
"""

import streamlit as st
from streamlit_extras.switch_page_button import switch_page

def main():
    # Main title and introduction
    st.title("📈 Forecasting Workflow")
    st.markdown("""
    Welcome to the **Forecasting Workflow**! Follow the steps below to generate stock price forecasts:
    
    ### Steps:
    1. **Fetch Raw Data:** Retrieve stock data from Yahoo Finance.
    2. **Clean Data:** Preprocess the data to prepare it for modeling.
    3. **Train Prophet Model:** Train a forecasting model using Facebook Prophet.
    4. **Forecast:** Generate future forecasts and visualize them interactively.
    """)

    # Horizontal rule for separation
    st.markdown("---")
    st.markdown("### Select a Step to Proceed:")

    # Navigation buttons for steps under `forecasting_steps`
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Step 1: Fetch Raw Data"):
            switch_page("forecasting_steps/01-Fetch_Raw_Data")
        if st.button("Step 3: Train Prophet Model"):
            switch_page("forecasting_steps/03-Train_Prophet")
    with col2:
        if st.button("Step 2: Clean Data"):
            switch_page("forecasting_steps/02-Clean_Data")
        if st.button("Step 4: Forecast"):
            switch_page("forecasting_steps/04-Forecast")

if __name__ == "__main__":
    main()
