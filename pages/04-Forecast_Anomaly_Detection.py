#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:22:08 2024

@author: thodoreskourtales
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fftpack import fft

def forecast_prices(model, holidays, periods):
    try:
        future = model.make_future_dataframe(periods=periods)
        future['holiday_effects'] = future['ds'].apply(lambda x: 1 if x in holidays['ds'].values else 0)
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
    try:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot historical data
        ax.plot(data['ds'], data['y'], label='Historical Stock Price', color='blue')

        # Plot forecast data
        forecast_positive = forecast[forecast['yhat'] > 0]
        ax.plot(forecast_positive['ds'], forecast_positive['yhat'], label='Forecasted Price', color='red')
        ax.fill_between(forecast_positive['ds'], forecast_positive['yhat_lower'], forecast_positive['yhat_upper'], color='pink', alpha=0.5)

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Forecast for {symbol}')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error plotting forecast for {symbol}: {e}")

def compute_rsi(series, window=14):
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        st.error(f"Error computing RSI: {e}")
        return pd.Series()

def compute_bollinger_bands(series, window=20):
    try:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        bollinger_upper = rolling_mean + (rolling_std * 2)
        bollinger_lower = rolling_mean - (rolling_std * 2)
        return bollinger_upper, bollinger_lower
    except Exception as e:
        st.error(f"Error computing Bollinger Bands: {e}")
        return pd.Series(), pd.Series()

def create_features(stock_data):
    try:
        stock_data['log_return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        stock_data['rolling_mean_5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['rolling_mean_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['rolling_std_5'] = stock_data['Close'].rolling(window=5).std()
        stock_data['rolling_std_20'] = stock_data['Close'].rolling(window=20).std()
        stock_data['ema_5'] = stock_data['Close'].ewm(span=5, adjust=False).mean()
        stock_data['ema_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
        stock_data['macd'] = stock_data['ema_5'] - stock_data['ema_20']
        stock_data['rsi'] = compute_rsi(stock_data['Close'])
        stock_data['bollinger_upper'], stock_data['bollinger_lower'] = compute_bollinger_bands(stock_data['Close'])
        stock_data['volume_rolling_mean'] = stock_data['Volume'].rolling(window=5).mean()
        stock_data['skewness'] = stock_data['Close'].rolling(window=20).skew()
        stock_data['kurtosis'] = stock_data['Close'].rolling(window=20).kurt()
        stock_data['z_score'] = (stock_data['Close'] - stock_data['Close'].rolling(window=20).mean()) / stock_data['Close'].rolling(window=20).std()
        stock_data['iqr'] = stock_data['Close'].rolling(window=20).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

        # Ensure the data is properly aligned and handle the FFT
        stock_data['fft'] = np.abs(fft(stock_data['Close'].fillna(0).values))

        # Add Seasonal Decomposition features
        decomposition = seasonal_decompose(stock_data['Close'].dropna(), model='additive', period=20)
        stock_data['seasonal'] = decomposition.seasonal
        stock_data['trend'] = decomposition.trend
        stock_data['resid'] = decomposition.resid

        stock_data.dropna(inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"Error creating features: {e}")
        return stock_data

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
    st.header("🔮 Step 4: Forecast")

    if 'prophet_model' not in st.session_state or 'holidays' not in st.session_state:
        st.warning("Prophet model not found. Please complete Step 3: Train Prophet Model.")
        return

    model = st.session_state['prophet_model']
    holidays = st.session_state['holidays']

    forecast_days = st.number_input("Number of days to forecast", min_value=1, max_value=365, value=30)
    forecast_button = st.button("Generate Forecast")

    if forecast_button:
        with st.spinner("Generating forecast..."):
            forecast = forecast_prices(model, holidays, forecast_days)
            if forecast is not None:
                st.success("Forecast generated successfully!")
                st.write("**Forecast Data Preview:**")
                st.dataframe(forecast.tail())

                # Plot Forecast
                plot_forecast_streamlit(st.session_state['cleaned_data'], forecast, st.session_state.get('symbol', 'Stock'))
            else:
                st.error("Forecast generation failed.")

if __name__ == "__main__":
    main()
