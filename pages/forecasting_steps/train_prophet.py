#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:21:23 2024

@author: thodoreskourtales
"""

# pages/3_Train_Prophet_Model.py

import streamlit as st
from prophet import Prophet
import pandas as pd
from streamlit_extras.switch_page_button import switch_page  # Import switch_page for navigation

def add_holiday_effects():
    """
    Define major US holidays to be included in the Prophet model.
    """
    holidays = pd.DataFrame({
        'holiday': ['New Year\'s Day', 'Independence Day', 'Christmas Day'],
        'ds': pd.to_datetime(['2023-01-01', '2023-07-04', '2023-12-25']),
        'lower_window': 0,
        'upper_window': 1,
    })
    return holidays

def train_prophet_model(data, holidays):
    """
    Train the Prophet model with the provided data and holidays.
    
    Parameters:
        data (pd.DataFrame): Cleaned stock data with 'ds' and 'y' columns.
        holidays (pd.DataFrame): DataFrame containing holiday information.
    
    Returns:
        model (Prophet): Trained Prophet model.
    """
    try:
        # Initialize Prophet with holidays
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            holidays=holidays
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit the model
        model.fit(data)
        
        return model
    except Exception as e:
        st.error(f"Error training Prophet model: {e}")
        return None

def main():
    st.header("📊 Step 3: Train Prophet Model")
    
    # Check if 'cleaned_data' and 'symbol' exist in session_state
    if 'cleaned_data' not in st.session_state or 'symbol' not in st.session_state:
        st.warning("No cleaned data or symbol found. Please complete Steps 1 & 2: Fetch and Clean Data.")
        return
    
    data = st.session_state['cleaned_data']
    symbol = st.session_state['symbol']
    
    st.subheader(f"Training Prophet Model for {symbol}")
    st.write("**Cleaned Data Preview:**")
    st.dataframe(data.head())
    
    train_button = st.button("Train Prophet Model")
    
    if train_button:
        with st.spinner("Training Prophet model..."):
            holidays = add_holiday_effects()
            model = train_prophet_model(data, holidays)
            if model:
                st.success("Prophet model trained successfully!")
                # Store the trained model and holidays in session_state
                st.session_state['prophet_model'] = model
                st.session_state['holidays'] = holidays
            else:
                st.error("Prophet model training failed.")
    
    # Navigation buttons
    st.markdown("---")
    


if __name__ == "__main__":
    main()
