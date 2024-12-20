#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:22:08 2024

@author: thodoreskourtales
"""

# pages/4_Forecast_Anomaly_Detection.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from scipy.fftpack import fft
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM
from keras.callbacks import EarlyStopping

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

def calculate_contamination(stock_data):
    try:
        q75, q25 = np.percentile(stock_data['Close'], [75, 25])
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        lower_bound = q25 - 1.5 * iqr
        outliers = stock_data[(stock_data['Close'] > upper_bound) | (stock_data['Close'] < lower_bound)]
        contamination_level = len(outliers) / len(stock_data)
        contamination_level = max(0.01, min(contamination_level, 0.5))  # Ensure contamination is within valid range
        return contamination_level
    except Exception as e:
        st.error(f"Error calculating contamination: {e}")
        return 0.05  # Default contamination level

def build_autoencoder(input_dim):
    try:
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(14, activation="tanh")(input_layer)
        encoder = Dense(7, activation="relu")(encoder)
        decoder = Dense(14, activation='tanh')(encoder)
        decoder = Dense(input_dim, activation='linear')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder
    except Exception as e:
        st.error(f"Error building autoencoder: {e}")
        return None

def train_autoencoder(autoencoder, X):
    try:
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        autoencoder.fit(X, X, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)
        return autoencoder
    except Exception as e:
        st.error(f"Error training autoencoder: {e}")
        return autoencoder

def fit_hmm(stock_data):
    try:
        model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100)
        features = ['Close', 'log_return', 'rolling_mean_5', 'rolling_mean_20', 'rolling_std_5', 'rolling_std_20',
                    'ema_5', 'ema_20', 'macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'volume_rolling_mean',
                    'skewness', 'kurtosis', 'z_score', 'iqr', 'fft', 'seasonal', 'trend', 'resid']
        X = stock_data[features].dropna().values
        model.fit(X)
        stock_data['hmm_anomaly'] = model.predict(X)
        stock_data['hmm_anomaly'] = stock_data['hmm_anomaly'].apply(lambda x: -1 if x == 0 else 1)  # Adjust labeling to match other methods
        return stock_data
    except Exception as e:
        st.error(f"Error fitting HMM: {e}")
        return stock_data

def fit_gmm(stock_data, contamination):
    try:
        features = ['Close', 'log_return', 'rolling_mean_5', 'rolling_mean_20', 'rolling_std_5', 'rolling_std_20',
                    'ema_5', 'ema_20', 'macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'volume_rolling_mean',
                    'skewness', 'kurtosis', 'z_score', 'iqr', 'fft', 'seasonal', 'trend', 'resid']
        X = stock_data[features].dropna().values
        model = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
        model.fit(X)
        scores = model.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        stock_data['gmm_anomaly'] = (scores < threshold).astype(int)
        stock_data['gmm_anomaly'] = stock_data['gmm_anomaly'].apply(lambda x: 1 if x == 1 else -1)  # Adjust labeling to match other methods
        return stock_data
    except Exception as e:
        st.error(f"Error fitting GMM: {e}")
        return stock_data

def detect_anomalies(stock_data, contamination):
    try:
        features = ['Close', 'log_return', 'rolling_mean_5', 'rolling_mean_20', 'rolling_std_5', 'rolling_std_20',
                    'ema_5', 'ema_20', 'macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'volume_rolling_mean',
                    'skewness', 'kurtosis', 'z_score', 'iqr', 'fft', 'seasonal', 'trend', 'resid']
        
        # Scale features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(stock_data[features])
        scaled_features_df = pd.DataFrame(scaled_features, columns=features)

        # Feature selection (optional)
        y = (stock_data['z_score'] > 2).astype(int)  # Example of how to get y for RFE
        model = RandomForestClassifier(n_estimators=100)
        rfe = RFE(model, n_features_to_select=10)
        fit = rfe.fit(scaled_features_df, y)
        selected_features = scaled_features_df.columns[fit.support_]
        scaled_features_selected = scaled_features_df[selected_features]

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(3, scaled_features_selected.shape[1]))  # keep at least 3 components or the max available
        pca_features = pca.fit_transform(scaled_features_selected)

        # Define anomaly detectors
        anomaly_detectors = {
            'Isolation Forest': IsolationForest(contamination=contamination, random_state=42, n_jobs=-1),
            'One-Class SVM': OneClassSVM(nu=contamination, kernel='rbf', gamma='scale'),
            'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5, n_jobs=-1),
            'OPTICS': OPTICS(min_samples=5, n_jobs=-1)
        }

        # Fit models and predict anomalies
        stock_data['Isolation Forest'] = anomaly_detectors['Isolation Forest'].fit_predict(scaled_features_selected)
        stock_data['One-Class SVM'] = anomaly_detectors['One-Class SVM'].fit_predict(scaled_features_selected)
        stock_data['Local Outlier Factor'] = anomaly_detectors['Local Outlier Factor'].fit_predict(scaled_features_selected)
        stock_data['DBSCAN'] = anomaly_detectors['DBSCAN'].fit_predict(pca_features)
        stock_data['OPTICS'] = anomaly_detectors['OPTICS'].fit_predict(pca_features)

        # Autoencoder anomaly detection
        autoencoder = build_autoencoder(scaled_features_selected.shape[1])
        if autoencoder:
            autoencoder = train_autoencoder(autoencoder, scaled_features_selected)
            if autoencoder:
                pred = autoencoder.predict(scaled_features_selected)
                mse = np.mean(np.power(scaled_features_selected - pred, 2), axis=1)
                threshold = np.percentile(mse, 100 * (1 - contamination))
                stock_data['Autoencoder'] = [1 if e > threshold else -1 for e in mse]
            else:
                stock_data['Autoencoder'] = -1
        else:
            stock_data['Autoencoder'] = -1

        # HMM anomaly detection
        stock_data = fit_hmm(stock_data)

        # GMM anomaly detection
        stock_data = fit_gmm(stock_data, contamination)

        return stock_data
    except Exception as e:
        st.error(f"Error during anomaly detection: {e}")
        return stock_data

def plot_anomalies_streamlit(stock_data, symbol):
    try:
        methods = ['Isolation Forest', 'One-Class SVM', 'Autoencoder', 
                   'Local Outlier Factor', 'DBSCAN', 'OPTICS', 
                   'hmm_anomaly', 'gmm_anomaly']
        titles = ['Isolation Forest', 'One-Class SVM', 'Autoencoder', 
                  'Local Outlier Factor', 'DBSCAN', 'OPTICS', 
                  'Hidden Markov Model', 'Gaussian Mixture Model']

        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        fig.suptitle(f'Types of Anomalies for {symbol}', fontsize=16)
        axes = axes.flatten()
        for i, method in enumerate(methods):
            ax = axes[i]
            ax.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue', alpha=0.3)
            if method in ['Autoencoder', 'Gaussian Mixture Model']:
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
    except Exception as e:
        st.error(f"Error plotting anomalies for {symbol}: {e}")

def summarize_anomalies(stock_data):
    try:
        summary = {
            'Isolation Forest': (stock_data['Isolation Forest'] == -1).sum(),
            'One-Class SVM': (stock_data['One-Class SVM'] == -1).sum(),
            'Autoencoder': (stock_data['Autoencoder'] == 1).sum(),
            'Local Outlier Factor': (stock_data['Local Outlier Factor'] == -1).sum(),
            'DBSCAN': (stock_data['DBSCAN'] == -1).sum(),
            'OPTICS': (stock_data['OPTICS'] == -1).sum(),
            'Hidden Markov Model': (stock_data['hmm_anomaly'] == -1).sum(),
            'Gaussian Mixture Model': (stock_data['gmm_anomaly'] == 1).sum()
        }
        return summary
    except Exception as e:
        st.error(f"Error summarizing anomalies: {e}")
        return {}

def main():
    st.header("🔮 Step 4: Forecast & Anomaly Detection")
    
    if 'prophet_model' not in st.session_state or 'holidays' not in st.session_state:
        st.warning("Prophet model not found. Please complete Step 3: Train Prophet Model.")
        return
    
    model = st.session_state['prophet_model']
    holidays = st.session_state['holidays']
    
    forecast_days = st.number_input("Number of days to forecast", min_value=1, max_value=365, value=30)
    forecast_button = st.button("Generate Forecast & Detect Anomalies")
    
    if forecast_button:
        with st.spinner("Generating forecast and detecting anomalies..."):
            forecast = forecast_prices(model, holidays, forecast_days)
            if forecast is not None:
                st.success("Forecast generated successfully!")
                st.write("**Forecast Data Preview:**")
                st.dataframe(forecast.tail())
                
                # Plot Forecast
                plot_forecast_streamlit(st.session_state['cleaned_data'], forecast, "Stock")
                
                # Anomaly Detection
                # Reload original stock data
                stock_data = st.session_state['cleaned_data'].copy()
                stock_data = stock_data.rename(columns={'ds': 'Date'})  # Ensure date column is named 'Date' for consistency
                
                stock_data = create_features(stock_data)
                contamination = calculate_contamination(stock_data)
                stock_data = detect_anomalies(stock_data, contamination)
                
                # Plot Anomalies
                st.subheader("📉 Detected Anomalies")
                plot_anomalies_streamlit(stock_data, "Stock")
                
                # Summarize Anomalies
                st.subheader("📄 Summary of Detected Anomalies")
                summary = summarize_anomalies(stock_data)
                summary_df = pd.DataFrame(list(summary.items()), columns=['Method', 'Anomaly Count'])
                st.table(summary_df)
            else:
                st.error("Forecast generation failed.")

if __name__ == "__main__":
    main()
