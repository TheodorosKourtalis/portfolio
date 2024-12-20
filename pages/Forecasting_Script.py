#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 23:07:02 2024

@author: thodoreskourtales
"""

import streamlit as st

def main():
    st.title("📈 Forecasting Workflow")
    st.markdown("""
    Welcome to the **Forecasting Workflow**! Follow the steps below to generate your stock price forecast:
    
    1. **Fetch Raw Data:** Retrieve stock data from Yahoo Finance.
    2. **Clean Data:** Preprocess the data to prepare it for modeling.
    3. **Train Prophet Model:** Train a forecasting model using Facebook Prophet.
    4. **Forecast:** Generate future forecasts and visualize them interactively.
    """)
    
    st.markdown("---")
    st.markdown("### Navigate to the Steps Below:")
    
    st.markdown("- [Step 1: Fetch Raw Data](1_Fetch_Raw_Data.py)")
    st.markdown("- [Step 2: Clean Data](2_Clean_Data.py)")
    st.markdown("- [Step 3: Train Prophet Model](3_Train_Prophet_Model.py)")
    st.markdown("- [Step 4: Forecast](4_Forecast.py)")
    
    st.markdown("---")
    st.markdown("**Start by clicking on Step 1 above or use the sidebar navigation.**")

if __name__ == "__main__":
    main()
