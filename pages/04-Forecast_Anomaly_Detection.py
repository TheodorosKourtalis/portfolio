#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:22:08 2024

@author: thodoreskourtales
"""

# pages/4_Forecast.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime

def forecast_prices(model, periods):
    """
    Generate future forecasts using the trained Prophet model.
    
    Parameters:
        model (Prophet): Trained Prophet model.
        periods (int): Number of days to forecast.
    
    Returns:
        forecast (pd.DataFrame): Forecasted data.
    """
    try:
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Ensure forecast values are not negative
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
        forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(x, 0))
        forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: max(x, 0))
        
        return forecast
    except Exception as e:
        st.error(f"Error during price forecasting: {e}")
        return None

def plot_forecast_streamlit(data, forecast, symbol):
    """
    Plot the historical and forecasted stock prices using Plotly.
    
    Parameters:
        data (pd.DataFrame): Historical stock data.
        forecast (pd.DataFrame): Forecasted stock data.
        symbol (str): Stock symbol.
    """
    try:
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=data['ds'],
            y=data['y'],
            mode='lines',
            name='Historical Stock Price',
            line=dict(color='blue')
        ))
        
        # Plot forecasted data
        forecast_positive = forecast[forecast['yhat'] > 0]
        fig.add_trace(go.Scatter(
            x=forecast_positive['ds'],
            y=forecast_positive['yhat'],
            mode='lines',
            name='Forecasted Price',
            line=dict(color='red')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_positive['ds'],
            y=forecast_positive['yhat_upper'],
            mode='lines',
            name='Upper Confidence Interval',
            line=dict(color='lightcoral'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_positive['ds'],
            y=forecast_positive['yhat_lower'],
            mode='lines',
            name='Lower Confidence Interval',
            line=dict(color='lightcoral'),
            fill='tonexty',
            fillcolor='rgba(255, 182, 193, 0.2)',
            showlegend=False
        ))
        
        # Update layout for better aesthetics
        fig.update_layout(
            title=f'Forecast for {symbol}',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(x=0, y=1),
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting forecast for {symbol}: {e}")

def main():
    st.header("🔮 Step 4: Forecast")
    
    # Check if required session states are present
    required_keys = ['prophet_model', 'holidays', 'cleaned_data', 'symbol']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    if missing_keys:
        st.warning(f"Missing data in session state: {', '.join(missing_keys)}. Please complete the previous steps.")
        if st.button("Go to Step 1: Fetch Raw Data"):
            from streamlit_extras.switch_page_button import switch_page
            switch_page("fetch raw data")
        return
    
    model = st.session_state['prophet_model']
    holidays = st.session_state['holidays']  # Not used here but kept for consistency
    cleaned_data = st.session_state['cleaned_data']
    symbol = st.session_state['symbol']
    
    st.subheader(f"Generating Forecast for {symbol}")
    
    forecast_days = st.number_input("Number of days to forecast", min_value=1, max_value=365, value=30)
    
    # Enhanced Decorative Arrows
    st.markdown(
        """<hr style="height:2px; border:none; background:linear-gradient(to right, #ff7e5f, #feb47b); margin: 10px 0;">""",
        unsafe_allow_html=True,
    )
    
    forecast_button = st.button("Generate Forecast")
    
    if forecast_button:
        with st.spinner("Generating forecast..."):
            forecast = forecast_prices(model, forecast_days)
            if forecast is not None:
                st.success("Forecast generated successfully!")
                
                # Enhanced Decorative Divider
                st.markdown(
                    """<hr style="height:2px; border:none; background:linear-gradient(to right, #6a11cb, #2575fc); margin: 10px 0;">""",
                    unsafe_allow_html=True,
                )
                
                st.write("**Forecast Data Preview:**")
                st.dataframe(forecast.tail())
                
                # Plot Forecast using Plotly
                plot_forecast_streamlit(cleaned_data, forecast, symbol)
            else:
                st.error("Forecast generation failed.")

if __name__ == "__main__":
    main()
