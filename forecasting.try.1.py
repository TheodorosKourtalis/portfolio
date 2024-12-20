#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit App for Stock Forecasting and Anomaly Detection

Created on Tue Jul 16 03:21:51 2024

@author: thodoreskourtales
"""

import subprocess
import sys
import importlib
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import logging
import streamlit as st

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fftpack import fft
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure required packages are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "numpy", "pandas", "scipy", "yfinance", "matplotlib", "scikit-learn",
    "prophet", "seaborn", "hmmlearn", "keras", "plotly", "statsmodels", "streamlit"
]
for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        install(package)

# **Set Streamlit page config before any other Streamlit commands**
st.set_page_config(page_title="Stock Forecasting & Anomaly Detection", layout="wide")

# Define the custom logging handler
class StreamlitHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        if 'logs' not in st.session_state:
            st.session_state['logs'] = []

    def emit(self, record):
        msg = self.format(record)
        st.session_state['logs'].append(msg)

# Set up logging with the custom handler
streamlit_handler = StreamlitHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
streamlit_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(streamlit_handler)
logger.setLevel(logging.INFO)

# Streamlit App
def main():
    st.title("📈 Stock Forecasting & Anomaly Detection App")
    st.markdown("""
        This application allows you to forecast stock prices using Prophet and detect anomalies using various algorithms.
    """)

    # Sidebar for user inputs
    st.sidebar.header("User Inputs")
    symbols_input = st.sidebar.text_input("Enter stock symbols (comma separated)", "AAPL, MSFT, GOOGL")
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    periods = st.sidebar.number_input("Number of days to forecast", min_value=1, max_value=365, value=30)
    run_button = st.sidebar.button("Run Forecast & Anomaly Detection")

    # Placeholder for logs
    log_placeholder = st.empty()

    if run_button:
        symbols = [symbol.strip().upper() for symbol in symbols_input.split(',')]
        for symbol in symbols:
            st.header(f"🔍 Processing {symbol}...")
            
            # Fetch data
            data = fetch_stock_data(symbol, start_date, end_date)
            if data is None or len(data) < 365:
                st.warning(f"Not enough data for {symbol}. Skipping...")
                continue

            # Preprocess data
            data = preprocess_data(data)
            if data is None or data.empty:
                st.warning(f"Preprocessed data is empty for {symbol}. Skipping...")
                continue

            # Train Prophet model
            model, holidays = train_prophet_model(data)
            if model is None:
                st.error(f"Failed to train Prophet model for {symbol}. Skipping...")
                continue

            # Forecast
            forecast = forecast_prices(model, holidays, periods)
            if forecast is None:
                st.error(f"Failed to forecast prices for {symbol}. Skipping...")
                continue

            # Fetch actual data for the forecasted period
            actual_end_date = (end_date + timedelta(days=periods)).strftime('%Y-%m-%d')
            actual_data = fetch_stock_data(symbol, end_date, actual_end_date)
            actual_data = preprocess_data(actual_data) if actual_data is not None else None

            # Plot forecast
            st.subheader("📊 Forecasted Prices")
            plot_forecast_streamlit(data, forecast, symbol, actual_data)

            # Evaluate forecast if actual data is available
            if actual_data is not None and not actual_data.empty:
                st.subheader("📈 Forecast Evaluation")
                evaluate_forecast_streamlit(actual_data['y'], forecast['yhat'][-len(actual_data):])

            # Log and calculate differences
            differences = log_and_calculate_differences(data, forecast, actual_data)

            # Plot differences
            last_historical_price = data['y'].iloc[-1]
            st.subheader("🔍 Price Differences")
            plot_differences_streamlit(differences, symbol, last_historical_price)

            # Anomaly detection
            st.subheader("🚨 Anomaly Detection")
            stock_data = load_stock_data(symbol, start_date, end_date)
            if stock_data.empty:
                st.error("No stock data to process. Skipping anomaly detection.")
                continue

            stock_data = create_features(stock_data)
            contamination = calculate_contamination(stock_data)
            stock_data = detect_anomalies(stock_data, contamination)

            # Plot anomalies
            st.subheader("📉 Detected Anomalies")
            plot_anomalies_streamlit(stock_data, symbol)

            # Summarize anomalies
            summary = summarize_anomalies(stock_data)
            st.subheader("📄 Summary of Detected Anomalies")
            summary_df = pd.DataFrame(list(summary.items()), columns=['Method', 'Anomaly Count'])
            st.table(summary_df)

        # Update and display logs
        if 'logs' in st.session_state:
            log_placeholder.text("\n".join(st.session_state['logs']))

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by [Thodores Kourtales](#)")

def fetch_stock_data(symbol, start_date, end_date):
    logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    try:
        # Convert dates to strings in 'YYYY-MM-DD' format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        stock_data = yf.download(symbol, start=start_str, end=end_str)
        if stock_data.empty:
            logging.warning(f"No data found for {symbol} between {start_str} and {end_str}.")
            return None
        logging.info(f"Successfully fetched data for {symbol}")
        logging.info(f"Data columns fetched: {stock_data.columns.tolist()}")
        return stock_data[['Close']].reset_index()
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

def preprocess_data(data):
    logging.info("Preprocessing data...")
    logging.info(f"Data columns before renaming: {data.columns.tolist()}")

    # Handle 'Date' or 'Datetime' columns
    if 'Date' in data.columns:
        date_col = 'Date'
    elif 'Datetime' in data.columns:
        date_col = 'Datetime'
    else:
        logging.error("Data does not contain required 'Date' or 'Datetime' columns.")
        return None

    # Ensure 'Close' column exists
    if 'Close' not in data.columns:
        logging.error("Data does not contain required 'Close' column.")
        return None

    # Rename columns
    data = data.rename(columns={date_col: 'ds', 'Close': 'y'})
    logging.info(f"Renamed '{date_col}' to 'ds' and 'Close' to 'y'.")

    # Replace zeros in 'y' with NaN and forward fill
    data['y'] = data['y'].replace(0, np.nan).ffill()
    logging.info("Replaced zeros in 'y' with NaN and forward filled.")

    # Check if 'y' and 'ds' columns are present after renaming
    if 'y' not in data.columns or 'ds' not in data.columns:
        logging.error("'y' or 'ds' columns are missing after renaming.")
        return None

    # Drop any remaining NaN values in 'y' and 'ds'
    data = data.dropna(subset=['y', 'ds'])
    logging.info("Dropped rows with NaN in 'y' or 'ds'.")

    # Ensure 'y' is of numeric type
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data = data.dropna(subset=['y'])
    logging.info("Converted 'y' to numeric and dropped any remaining NaN values.")

    # Debugging: Log the first few rows and data types
    logging.info(f"Preprocessed data head:\n{data.head()}")
    logging.info(f"Data types:\n{data.dtypes}")

    logging.info("Data preprocessing complete.")
    return data

def add_holiday_effects(model, data):
    logging.info("Adding holiday effects to the model...")
    try:
        holidays = pd.DataFrame({
            'holiday': 'major_holidays',
            'ds': pd.to_datetime(['2023-01-01', '2023-07-04', '2023-12-25']),
            'lower_window': 0,
            'upper_window': 1,
        })
        model.add_country_holidays(country_name='US')
        logging.info("Added US country holidays to the model.")
        
        # Ensure 'ds' column exists in data
        if 'ds' not in data.columns:
            logging.error("'ds' column not found in data. Cannot add holiday effects.")
            return model, data, holidays
        
        data['holiday_effects'] = data['ds'].apply(lambda x: 1 if x in holidays['ds'].values else 0)
        logging.info("Holiday effects added to the data.")
        
        return model, data, holidays
    except Exception as e:
        logging.error(f"Error adding holiday effects: {e}")
        return model, data, holidays

def train_prophet_model(data):
    logging.info("Training Prophet model...")
    try:
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        logging.info("Initialized Prophet model with custom monthly seasonality.")
        
        model, data, holidays = add_holiday_effects(model, data)
        
        # Check if 'ds' and 'y' columns are present
        if 'ds' not in data.columns or 'y' not in data.columns:
            logging.error("'ds' or 'y' columns are missing. Cannot fit Prophet model.")
            return None, None
        
        model.fit(data)
        logging.info("Prophet model training complete.")
        return model, holidays
    except Exception as e:
        logging.error(f"Error training Prophet model: {e}")
        return None, None

def forecast_prices(model, holidays, periods):
    logging.info(f"Forecasting prices for the next {periods} days...")
    try:
        future = model.make_future_dataframe(periods=periods)
        future['holiday_effects'] = future['ds'].apply(lambda x: 1 if x in holidays['ds'].values else 0)
        forecast = model.predict(future)
        
        # Ensure forecast values are not negative
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
        forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(x, 0))
        forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: max(x, 0))
        
        logging.info("Price forecasting complete.")
        return forecast
    except Exception as e:
        logging.error(f"Error during price forecasting: {e}")
        return None

def plot_forecast_streamlit(data, forecast, symbol, actual_data=None):
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot historical data
        ax.plot(data['ds'], data['y'], label='Historical Stock Price', color='blue')
        
        # Plot forecast data
        forecast_positive = forecast[forecast['yhat'] > 0]
        ax.plot(forecast_positive['ds'], forecast_positive['yhat'], label='Forecasted Price', color='red')
        ax.fill_between(forecast_positive['ds'], forecast_positive['yhat_lower'], forecast_positive['yhat_upper'], color='pink', alpha=0.5)
        
        # Plot actual future data if available
        if actual_data is not None:
            ax.plot(actual_data['ds'], actual_data['y'], label='Actual Future Price', color='green')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Forecast for {symbol}')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)
        logging.info(f"Plotting complete for {symbol}.")
    except Exception as e:
        logging.error(f"Error plotting forecast for {symbol}: {e}")

def evaluate_forecast_streamlit(actual, predicted):
    try:
        if len(actual) < len(predicted):
            predicted = predicted[-len(actual):]
        elif len(actual) > len(predicted):
            actual = actual[-len(predicted):]
            
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = mse ** 0.5
        logging.info(f'Mean Absolute Error (MAE): {mae:.4f}')
        logging.info(f'Mean Squared Error (MSE): {mse:.4f}')
        logging.info(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
        
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    except Exception as e:
        logging.error(f"Error evaluating forecast: {e}")
        st.error(f"Error evaluating forecast: {e}")

def log_and_calculate_differences(data, forecast, actual_data=None):
    differences = {}
    try:
        last_historical_price = data['y'].iloc[-1]
        last_forecasted_price = forecast['yhat'].iloc[-1]
        
        # Log the prices for verification
        logging.info(f"Last historical price: {last_historical_price:.2f}")
        logging.info(f"Last forecasted price: {last_forecasted_price:.2f}")

        # Calculate the differences
        difference_forecasted_future_historical = last_forecasted_price - last_historical_price
        differences["Forecasted vs Historical"] = difference_forecasted_future_historical
        logging.info(f"Difference between last forecasted price and last historical price: {difference_forecasted_future_historical:.2f}")

        # Calculate and log differences if actual future data is available
        if actual_data is not None and not actual_data.empty:
            last_actual_future_price = actual_data['y'].iloc[-1]
            
            # Find the closest matching date in the forecast data
            closest_date = actual_data['ds'].iloc[-1]
            closest_forecast_row = forecast.iloc[(forecast['ds'] - closest_date).abs().argsort()[:1]]
            closest_forecasted_price = closest_forecast_row['yhat'].values[0]
            
            logging.info(f"Last actual future price: {last_actual_future_price:.2f}")
            logging.info(f"Closest forecasted price: {closest_forecasted_price:.2f}")

            difference_actual_future_historical = last_actual_future_price - last_historical_price
            difference_forecasted_actual = closest_forecasted_price - last_actual_future_price

            differences["Actual Future vs Historical"] = difference_actual_future_historical
            differences["Forecasted vs Actual Future"] = difference_forecasted_actual

            logging.info(f"Difference between last actual future price and last historical price: {difference_actual_future_historical:.2f}")
            logging.info(f"Difference between closest forecasted price and last actual future price: {difference_forecasted_actual:.2f}")

    except Exception as e:
        logging.error(f"Error calculating differences: {e}")
    
    return differences

def plot_differences_streamlit(differences, symbol, last_historical_price):
    try:
        labels = list(differences.keys())
        values = list(differences.values())

        # Use the absolute value of the last historical price to calculate percentage differences
        base_value = abs(last_historical_price)
        percentages = [(value / base_value) * 100 if base_value != 0 else 0 for value in values]

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot absolute differences
        axs[0].barh(labels, values, color='skyblue')
        axs[0].set_xlabel('Price Difference')
        axs[0].set_title(f'Price Differences for {symbol}')
        
        # Plot percentage differences
        axs[1].barh(labels, percentages, color='salmon')
        axs[1].set_xlabel('Percentage Difference (%)')
        axs[1].set_title(f'Percentage Differences for {symbol}')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Log percentage differences to the console
        logging.info(f"Percentage Differences for {symbol}:")
        for label, percentage in zip(labels, percentages):
            logging.info(f"{label}: {percentage:.2f}%")
    except Exception as e:
        logging.error(f"Error plotting differences for {symbol}: {e}")

def load_stock_data(stock_name, start_date='2020-01-01', end_date=None):
    end_date = end_date or pd.Timestamp.today().strftime('%Y-%m-%d')
    try:
        # Convert dates to strings in 'YYYY-MM-DD' format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        stock_data = yf.download(stock_name, start=start_str, end=end_str)
        if stock_data.empty:
            logging.warning("No data fetched for the given stock symbol.")
            return pd.DataFrame()
        stock_data.reset_index(inplace=True)
        logging.info("Stock data loaded successfully.")
        return stock_data
    except Exception as e:
        logging.error(f"Error loading stock data: {e}")
        return pd.DataFrame()

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
        logging.info("Features created successfully.")
        return stock_data
    except Exception as e:
        logging.error(f"Error creating features: {e}")
        return stock_data

def compute_rsi(series, window=14):
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logging.error(f"Error computing RSI: {e}")
        return pd.Series()

def compute_bollinger_bands(series, window=20):
    try:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        bollinger_upper = rolling_mean + (rolling_std * 2)
        bollinger_lower = rolling_mean - (rolling_std * 2)
        return bollinger_upper, bollinger_lower
    except Exception as e:
        logging.error(f"Error computing Bollinger Bands: {e}")
        return pd.Series(), pd.Series()

def build_autoencoder(input_dim):
    try:
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(14, activation="tanh")(input_layer)
        encoder = Dense(7, activation="relu")(encoder)
        decoder = Dense(14, activation='tanh')(encoder)
        decoder = Dense(input_dim, activation='linear')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("Autoencoder model built successfully.")
        return autoencoder
    except Exception as e:
        logging.error(f"Error building autoencoder: {e}")
        return None

def train_autoencoder(autoencoder, X):
    try:
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        autoencoder.fit(X, X, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)
        logging.info("Autoencoder trained successfully.")
        return autoencoder
    except Exception as e:
        logging.error(f"Error training autoencoder: {e}")
        return autoencoder

def build_lstm_autoencoder(input_shape):
    try:
        model = Sequential()
        model.add(LSTM(128, activation='relu', input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(input_shape[2], activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        logging.info("LSTM Autoencoder model built successfully.")
        return model
    except Exception as e:
        logging.error(f"Error building LSTM Autoencoder: {e}")
        return None

def fit_hmm(stock_data):
    try:
        model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100)
        features = [
            'Close', 'log_return', 'rolling_mean_5', 'rolling_mean_20', 'rolling_std_5', 'rolling_std_20',
            'ema_5', 'ema_20', 'macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'volume_rolling_mean',
            'skewness', 'kurtosis', 'z_score', 'iqr', 'fft', 'seasonal', 'trend', 'resid'
        ]
        X = stock_data[features].dropna().values
        model.fit(X)
        stock_data['hmm_anomaly'] = model.predict(X)
        stock_data['hmm_anomaly'] = stock_data['hmm_anomaly'].apply(lambda x: -1 if x == 0 else 1)  # Adjust labeling to match other methods
        logging.info("HMM anomaly detection completed.")
        return stock_data
    except Exception as e:
        logging.error(f"Error fitting HMM: {e}")
        return stock_data

def fit_gmm(stock_data, contamination):
    try:
        features = [
            'Close', 'log_return', 'rolling_mean_5', 'rolling_mean_20', 'rolling_std_5', 'rolling_std_20',
            'ema_5', 'ema_20', 'macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'volume_rolling_mean',
            'skewness', 'kurtosis', 'z_score', 'iqr', 'fft', 'seasonal', 'trend', 'resid'
        ]
        X = stock_data[features].dropna().values
        model = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
        model.fit(X)
        scores = model.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        stock_data['gmm_anomaly'] = (scores < threshold).astype(int)
        stock_data['gmm_anomaly'] = stock_data['gmm_anomaly'].apply(lambda x: 1 if x == 1 else -1)  # Adjust labeling to match other methods
        logging.info("GMM anomaly detection completed.")
        return stock_data
    except Exception as e:
        logging.error(f"Error fitting GMM: {e}")
        return stock_data

def calculate_contamination(stock_data):
    try:
        q75, q25 = np.percentile(stock_data['Close'], [75, 25])
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        lower_bound = q25 - 1.5 * iqr
        outliers = stock_data[(stock_data['Close'] > upper_bound) | (stock_data['Close'] < lower_bound)]
        contamination_level = len(outliers) / len(stock_data)
        logging.info(f"Outliers found: {len(outliers)}, Total data points: {len(stock_data)}")
        logging.info(f"Calculated contamination level before adjustment: {contamination_level:.2%}")
        contamination_level = max(0.01, min(contamination_level, 0.5))  # Ensure contamination is within valid range
        logging.info(f"Contamination level after adjustment: {contamination_level:.2%}")
        return contamination_level
    except Exception as e:
        logging.error(f"Error calculating contamination: {e}")
        return 0.05  # Default contamination level

def select_features(X, y):
    try:
        model = RandomForestClassifier(n_estimators=100)
        rfe = RFE(model, n_features_to_select=10)
        fit = rfe.fit(X, y)
        selected_features = X.columns[fit.support_]
        logging.info(f"Selected features: {list(selected_features)}")
        return selected_features
    except Exception as e:
        logging.error(f"Error selecting features: {e}")
        return X.columns  # Return all features if selection fails

def tune_hyperparameters(model, X, y):
    try:
        param_grid = {
            'n_estimators': [100, 200],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        logging.info(f"Best parameters found: {grid_search.best_params_}")
        return grid_search.best_params_
    except Exception as e:
        logging.error(f"Error tuning hyperparameters: {e}")
        return {}

def scale_features(X):
    try:
        scaler = RobustScaler()
        scaled_X = scaler.fit_transform(X)
        logging.info("Features scaled using RobustScaler.")
        return scaled_X
    except Exception as e:
        logging.error(f"Error scaling features: {e}")
        return X

def evaluate_model(y_true, y_pred):
    try:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        logging.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")

def make_stationary(series):
    try:
        stationary_series = series.diff().dropna()
        logging.info("Converted series to stationary.")
        return stationary_series
    except Exception as e:
        logging.error(f"Error making series stationary: {e}")
        return series

def detect_anomalies(stock_data, contamination):
    try:
        features = [
            'Close', 'log_return', 'rolling_mean_5', 'rolling_mean_20', 'rolling_std_5', 'rolling_std_20',
            'ema_5', 'ema_20', 'macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'volume_rolling_mean',
            'skewness', 'kurtosis', 'z_score', 'iqr', 'fft', 'seasonal', 'trend', 'resid'
        ]
        
        # Scale features
        scaled_features = scale_features(stock_data[features])
        scaled_features_df = pd.DataFrame(scaled_features, columns=features)

        # Feature selection
        y = (stock_data['z_score'] > 2).astype(int)  # Example of how to get y for RFE
        selected_features = select_features(scaled_features_df, y)
        scaled_features_selected = scaled_features_df[selected_features]

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(3, scaled_features_selected.shape[1]))  # keep at least 3 components or the max available
        pca_features = pca.fit_transform(scaled_features_selected)
        logging.info("Applied PCA for dimensionality reduction.")

        # Define anomaly detectors
        anomaly_detectors = {
            'if': IsolationForest(contamination=contamination, random_state=42, n_jobs=-1),
            'svm': OneClassSVM(nu=contamination, kernel='rbf', gamma='scale'),
            'lof': LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1),
            'dbscan': DBSCAN(eps=0.5, min_samples=5, n_jobs=-1),
            'optics': OPTICS(min_samples=5, n_jobs=-1)
        }

        # Convert back to DataFrame to retain feature names
        scaled_features_selected_df = pd.DataFrame(scaled_features_selected, columns=selected_features)

        # Fit models and predict anomalies
        stock_data['if_anomaly'] = anomaly_detectors['if'].fit_predict(scaled_features_selected_df)
        stock_data['svm_anomaly'] = anomaly_detectors['svm'].fit_predict(scaled_features_selected_df)
        stock_data['lof_anomaly'] = anomaly_detectors['lof'].fit_predict(scaled_features_selected_df)
        stock_data['dbscan_anomaly'] = anomaly_detectors['dbscan'].fit_predict(pca_features)
        stock_data['optics_anomaly'] = anomaly_detectors['optics'].fit_predict(pca_features)

        # Autoencoder anomaly detection
        autoencoder = build_autoencoder(scaled_features_selected_df.shape[1])
        if autoencoder:
            autoencoder = train_autoencoder(autoencoder, scaled_features_selected_df)
            if autoencoder:
                pred = autoencoder.predict(scaled_features_selected_df)
                mse = np.mean(np.power(scaled_features_selected_df - pred, 2), axis=1)
                threshold = np.percentile(mse, 100 * (1 - contamination))
                stock_data['autoencoder_anomaly'] = [1 if e > threshold else -1 for e in mse]
                logging.info("Autoencoder anomaly detection completed.")
            else:
                stock_data['autoencoder_anomaly'] = -1
        else:
            stock_data['autoencoder_anomaly'] = -1

        # HMM anomaly detection
        stock_data = fit_hmm(stock_data)

        # GMM anomaly detection
        stock_data = fit_gmm(stock_data, contamination)

        logging.info("Anomaly detection completed.")
        return stock_data
    except Exception as e:
        logging.error(f"Error during anomaly detection: {e}")
        return stock_data

def plot_anomalies_streamlit(stock_data, symbol):
    try:
        methods = [
            'if_anomaly', 'svm_anomaly', 'autoencoder_anomaly',
            'lof_anomaly', 'dbscan_anomaly', 'optics_anomaly',
            'hmm_anomaly', 'gmm_anomaly'
        ]
        titles = [
            'Isolation Forest', 'One-Class SVM', 'Autoencoder',
            'Local Outlier Factor', 'DBSCAN', 'OPTICS',
            'Hidden Markov Model', 'Gaussian Mixture Model'
        ]

        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        fig.suptitle(f'Types of Anomalies for {symbol}', fontsize=16)
        axes = axes.flatten()
        for i, method in enumerate(methods):
            ax = axes[i]
            ax.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue', alpha=0.3)
            if method in ['autoencoder_anomaly', 'gmm_anomaly']:
                anomalies = stock_data[stock_data[method] == 1]
            else:
                anomalies = stock_data[stock_data[method] == -1]
            if not anomalies.empty:
                ax.scatter(anomalies['Date'], anomalies['Close'], color='red', label=f'{titles[i]} Anomalies', s=50)
            ax.set_title(f'{titles[i]} Anomalies')
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        st.pyplot(fig)
        plt.close(fig)
        logging.info(f"Anomaly plotting complete for {symbol}.")
    except Exception as e:
        logging.error(f"Error plotting anomalies for {symbol}: {e}")

def summarize_anomalies(stock_data):
    try:
        summary = {
            'Isolation Forest': (stock_data['if_anomaly'] == -1).sum(),
            'One-Class SVM': (stock_data['svm_anomaly'] == -1).sum(),
            'Autoencoder': (stock_data['autoencoder_anomaly'] == 1).sum(),
            'Local Outlier Factor': (stock_data['lof_anomaly'] == -1).sum(),
            'DBSCAN': (stock_data['dbscan_anomaly'] == -1).sum(),
            'OPTICS': (stock_data['optics_anomaly'] == -1).sum(),
            'HMM': (stock_data['hmm_anomaly'] == -1).sum(),
            'GMM': (stock_data['gmm_anomaly'] == 1).sum()
        }
        logging.info(f"Anomaly summary: {summary}")
        return summary
    except Exception as e:
        logging.error(f"Error summarizing anomalies: {e}")
        return {}

if __name__ == "__main__":
    main()
