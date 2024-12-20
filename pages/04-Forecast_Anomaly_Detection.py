#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:22:08 2024

@author: thodoreskourtales
"""

# pages/4_Forecast.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
from streamlit_extras.switch_page_button import switch_page  # Import switch_page for navigation

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

def plot_percentage_change(cleaned_data, forecast):
    """
    Plot the percentage change between historical and forecasted prices.
    
    Parameters:
        cleaned_data (pd.DataFrame): Cleaned historical stock data.
        forecast (pd.DataFrame): Forecasted stock data.
    """
    try:
        # Merge historical data with forecast to get 'y' values aligned with 'ds'
        merged = pd.merge(cleaned_data, forecast[['ds', 'yhat']], on='ds', how='left')
        
        # Calculate percentage change where 'y' exists
        merged['pct_change'] = ((merged['yhat'] - merged['y']) / merged['y']) * 100
        
        # Drop rows where 'y' is NaN (future forecasts)
        pct_change_data = merged.dropna(subset=['y'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=pct_change_data['ds'],
            y=pct_change_data['pct_change'],
            name='Percentage Change',
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='Percentage Change from Historical to Forecasted Prices',
            xaxis_title='Date',
            yaxis_title='Percentage Change (%)',
            template='plotly_white',
            width=1000,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting percentage change: {e}")

def main():
    st.header("🔮 Step 4: Forecast")
    
    # Check if required session states are present
    required_keys = ['prophet_model', 'holidays', 'cleaned_data', 'symbol']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    if missing_keys:
        st.warning(f"Missing data in session state: {', '.join(missing_keys)}. Please complete the previous steps.")
        if st.button("Go to Step 1: Fetch Raw Data"):
            switch_page("fetch raw data")
        return
    
    model = st.session_state['prophet_model']
    holidays = st.session_state['holidays']  # Not used here but kept for consistency
    cleaned_data = st.session_state['cleaned_data']
    symbol = st.session_state['symbol']
    
    st.subheader(f"Generating Forecast for {symbol}")
    
    # Number of days to forecast
    forecast_days = st.number_input("Number of days to forecast", min_value=1, max_value=365, value=30)
    
    # Select specific day within the forecast period
    specific_day = st.slider(
        "Select the day to view forecasted value:",
        min_value=1,
        max_value=forecast_days,
        value=15,
        step=1
    )
    
    # Stylish Pointer Before Forecast Generation
    st.markdown(
        """
        <div style="text-align:center; font-size: 20px; margin: 20px 0;">
            <span style="color: #555; font-weight: bold;">👇 Scroll Down to Generate Forecast 👇</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    forecast_button = st.button("Generate Forecast")
    
    if forecast_button:
        with st.spinner("Generating forecast..."):
            forecast = forecast_prices(model, forecast_days)
            if forecast is not None:
                st.success("Forecast generated successfully!")
                
                # Stylish Pointer After Success
                st.markdown(
                    """
                    <div style="text-align:center; font-size: 20px; margin: 20px 0;">
                        <span style="color: #0078FF; font-weight: bold;">📈 Scroll Down for Forecast Details 📈</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                st.write("**Forecast Data Preview:**")
                st.dataframe(forecast.tail())
                
                # Display forecasted value for the selected day
                if specific_day <= forecast_days:
                    forecast_date = forecast['ds'].iloc[-forecast_days + specific_day - 1]
                    forecast_value = forecast['yhat'].iloc[-forecast_days + specific_day - 1]
                    forecast_lower = forecast['yhat_lower'].iloc[-forecast_days + specific_day - 1]
                    forecast_upper = forecast['yhat_upper'].iloc[-forecast_days + specific_day - 1]
                    
                    st.markdown(
                        f"""
                        ### 📅 Forecast for {forecast_date.date()}:
                        - **Predicted Price:** ${forecast_value:,.2f}
                        - **Confidence Interval:** (${forecast_lower:,.2f}, ${forecast_upper:,.2f})
                        """
                    )
                else:
                    st.warning("Selected day exceeds the forecast period.")
                
                # Plot Forecast using Plotly
                plot_forecast_streamlit(cleaned_data, forecast, symbol)
                
                # Plot Percentage Change
                plot_percentage_change(cleaned_data, forecast)
            else:
                st.error("Forecast generation failed.")
    
    # Navigation buttons
    st.markdown("---")
    
    # Show "Next Step" button only if forecast is generated
    if 'prophet_model' in st.session_state and 'cleaned_data' in st.session_state:
        st.markdown("### Navigate to the Next Step:")
        if st.button("Next Step: Fetch Raw Data"):
            switch_page("fetch raw data")  # Redirect to Step 1 since there's no next step yet
    
    # Show "Previous Step" button
    st.markdown("### Return to the Previous Step:")
    if st.button("Previous Step: Train Prophet Model"):
        switch_page("train prophet")

if __name__ == "__main__":
    main()
