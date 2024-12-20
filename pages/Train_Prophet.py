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
def add_holiday_effects(model, data):
    holidays = pd.DataFrame({
        'holiday': 'major_holidays',
        'ds': pd.to_datetime(['2023-01-01', '2023-07-04', '2023-12-25']),
        'lower_window': 0,
        'upper_window': 1,
    })
    model.add_country_holidays(country_name='US')
    data['holiday_effects'] = data['ds'].apply(lambda x: 1 if x in holidays['ds'].values else 0)
    return model, data, holidays

def train_prophet_model(data):
    try:
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model, data, holidays = add_holiday_effects(model, data)
        model.fit(data)
        return model, holidays
    except Exception as e:
        st.error(f"Error training Prophet model: {e}")
        return None, None

def main():
    st.header("📊 Step 3: Train Prophet Model")
    
    if 'cleaned_data' not in st.session_state:
        st.warning("No cleaned data found. Please complete Step 2: Clean Data.")
        return
    
    data = st.session_state['cleaned_data']
    
    st.write("**Cleaned Data Preview:**")
    st.dataframe(data.head())
    
    train_button = st.button("Train Prophet Model")
    
    if train_button:
        with st.spinner("Training Prophet model..."):
            model, holidays = train_prophet_model(data)
            if model is not None:
                st.success("Prophet model trained successfully!")
                st.session_state['prophet_model'] = model
                st.session_state['holidays'] = holidays
            else:
                st.error("Prophet model training failed.")

if __name__ == "__main__":
    main()