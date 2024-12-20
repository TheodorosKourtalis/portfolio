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
    st.markdown("### Select a Step to Proceed:")
    
    # Navigation Buttons
    if st.button("Step 1: Fetch Raw Data"):
        st.session_state["page"] = "1_Fetch_Raw_Data"
    
    if st.button("Step 2: Clean Data"):
        st.session_state["page"] = "2_Clean_Data"
    
    if st.button("Step 3: Train Prophet Model"):
        st.session_state["page"] = "3_Train_Prophet_Model"
    
    if st.button("Step 4: Forecast"):
        st.session_state["page"] = "4_Forecast"

    # Display the current page state for debugging (optional)
    # st.write(f"Current Page: {st.session_state.get('page', 'None')}")

if __name__ == "__main__":
    main()
