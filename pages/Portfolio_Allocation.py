#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Optimization Dashboard - Single Page Streamlit App

Author: Theodoros Kourtales
Date: Fri Dec 20 22:20:28 2024
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from pypfopt import risk_models, expected_returns, EfficientFrontier
from pypfopt import plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import matplotlib.pyplot as plt
import io

# Suppress warnings for a cleaner interface
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(
    page_title="📈 Portfolio Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch adjusted close data for given tickers and date range.
    
    Parameters:
        tickers (list): List of stock ticker symbols.
        start_date (datetime): Start date for data fetching.
        end_date (datetime): End date for data fetching.
        
    Returns:
        adj_close (DataFrame): Adjusted close prices.
        valid_tickers (list): List of valid tickers fetched.
    """
    try:
        # Remove duplicates and ensure uppercase
        unique_tickers = list(dict.fromkeys([ticker.strip().upper() for ticker in tickers]))
        if not unique_tickers:
            st.error("❌ No ticker symbols provided.")
            return None, None
        
        # Fetch data using yfinance
        data = yf.download(unique_tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
        
        if data.empty:
            st.error("❌ No data fetched. Please check your ticker symbols and date range.")
            return None, None
        
        adj_close = pd.DataFrame()
        invalid_tickers = []
        
        for ticker in unique_tickers:
            try:
                if len(unique_tickers) == 1:
                    # Single ticker scenario
                    adj_close[ticker] = data['Adj Close']
                else:
                    # Multiple tickers scenario
                    adj_close[ticker] = data[ticker]['Adj Close']
            except KeyError:
                invalid_tickers.append(ticker)
        
        # Drop tickers with all NaN values
        adj_close.dropna(axis=1, how='all', inplace=True)
        
        # Identify remaining invalid tickers
        remaining_invalid = [ticker for ticker in unique_tickers if ticker not in adj_close.columns]
        if remaining_invalid:
            invalid_tickers.extend(remaining_invalid)
        
        if invalid_tickers:
            st.warning(f"⚠️ The following tickers were invalid or have no 'Adj Close' data and have been excluded: {', '.join(invalid_tickers)}")
        
        if adj_close.empty:
            st.error("❌ No valid tickers with 'Adj Close' data found.")
            return None, None
        
        return adj_close, unique_tickers
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return None, None

def calculate_daily_returns(adj_close):
    """
    Calculate daily percentage returns from adjusted close prices.
    
    Parameters:
        adj_close (DataFrame): Adjusted close prices.
        
    Returns:
        returns (DataFrame): Daily returns.
    """
    returns = adj_close.pct_change().dropna()
    return returns

def optimize_portfolio(returns, risk_free_rate=0.02):
    """
    Perform Mean-Variance Optimization to maximize Sharpe Ratio.
    
    Parameters:
        returns (DataFrame): Daily returns of the stocks.
        risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation.
        
    Returns:
        weights (dict): Optimized portfolio weights.
        exp_return (float): Expected annual return.
        vol (float): Annual volatility.
        sharpe (float): Sharpe Ratio.
    """
    mu = expected_returns.mean_historical_return(returns)
    S = risk_models.sample_cov(returns)
    
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()
    exp_return, vol, sharpe = ef.portfolio_performance(verbose=False)
    
    return cleaned_weights, exp_return, vol, sharpe

def plot_efficient_frontier(mu, S):
    """
    Plot the Efficient Frontier using Matplotlib and convert it to an image for Streamlit.
    
    Parameters:
        mu (Series): Expected returns.
        S (DataFrame): Covariance matrix.
        
    Returns:
        image (bytes): Image bytes of the Efficient Frontier plot.
    """
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    plt.title("📈 Efficient Frontier")
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image = buf.read()
    
    return image

# --- Main Application ---

def main():
    st.title("📈 Portfolio Optimization Dashboard")
    st.markdown("""
    This application fetches and analyzes financial data for specified companies. 
    Enter the ticker symbols, select the date range, and perform portfolio optimization 
    to explore various financial metrics and portfolio allocations.
    """)
    
    # Sidebar Inputs
    st.sidebar.header("🔧 Input Parameters")
    
    tickers_input = st.sidebar.text_input(
        "📊 Enter Ticker Symbols (comma-separated, e.g., AAPL, MSFT)",
        value="AAPL, MSFT"
    )
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    
    start_date = st.sidebar.date_input(
        "📅 Start Date",
        value=datetime(2020, 1, 1),
        min_value=datetime(1900, 1, 1),
        max_value=datetime.today()
    )
    
    end_date = st.sidebar.date_input(
        "📅 End Date",
        value=datetime.today(),
        min_value=start_date,
        max_value=datetime.today()
    )
    
    # Risk-Free Rate Input
    risk_free_rate = st.sidebar.number_input(
        "💵 Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Assumed risk-free rate for Sharpe Ratio calculations."
    ) / 100  # Convert to decimal
    
    # Fetch Data Button
    fetch_button = st.sidebar.button("🚀 Fetch Data")
    
    if fetch_button:
        with st.spinner("📥 Fetching market data..."):
            adj_close, valid_tickers = fetch_stock_data(tickers, start_date, end_date)
            if adj_close is not None:
                st.success("✅ Market data fetched successfully!")
                st.session_state['adj_close'] = adj_close
                st.session_state['valid_tickers'] = valid_tickers
            else:
                st.error("❌ Failed to fetch market data. Please check your ticker symbols and internet connection.")
    
    # Proceed only if data is fetched
    if 'adj_close' in st.session_state:
        adj_close = st.session_state['adj_close']
        valid_tickers = st.session_state['valid_tickers']
        
        st.header("🗃️ Fetched Data")
        st.write(f"**Tickers:** {', '.join(valid_tickers)}")
        st.write("**Raw Adjusted Close Data Preview:**")
        st.dataframe(adj_close.head())
        
        # Option to download raw data
        st.download_button(
            label="💾 Download Raw Data as CSV",
            data=adj_close.to_csv(index=True).encode('utf-8'),
            file_name="raw_stock_data.csv",
            mime="text/csv"
        )
        
        # Clean Data Section
        st.header("🧹 Clean Data")
        st.markdown("""
        The data has been fetched and is ready for cleaning. This involves handling missing values and preparing the data for analysis.
        """)
        
        # Calculate and display daily returns
        returns = calculate_daily_returns(adj_close)
        st.subheader("📊 Daily Returns")
        st.write("**Daily Returns Preview:**")
        st.dataframe(returns.head())
        
        # Option to download returns data
        st.download_button(
            label="💾 Download Daily Returns as CSV",
            data=returns.to_csv(index=True).encode('utf-8'),
            file_name="daily_returns.csv",
            mime="text/csv"
        )
        
        # Portfolio Allocation Section
        st.header("📈 Portfolio Allocation")
        st.markdown("""
        Perform portfolio optimization using the Mean-Variance Optimization technique to maximize the Sharpe Ratio.
        """)
        
        optimize_button = st.button("🛠️ Optimize Portfolio")
        
        if optimize_button:
            with st.spinner("🔍 Performing portfolio optimization..."):
                weights, exp_return, vol, sharpe = optimize_portfolio(returns, risk_free_rate=risk_free_rate)
                st.success("✅ Portfolio optimized successfully!")
                
                # Display optimized weights
                st.subheader("📊 Optimized Portfolio Weights")
                weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                weights_df = weights_df[weights_df['Weight'] > 0]
                weights_df = weights_df.sort_values(by='Weight', ascending=False)
                st.dataframe(weights_df)
                
                # Download optimized weights
                st.download_button(
                    label="💾 Download Optimized Weights as CSV",
                    data=weights_df.to_csv().encode('utf-8'),
                    file_name="optimized_weights.csv",
                    mime="text/csv"
                )
                
                # Display performance metrics
                st.subheader("📈 Portfolio Performance Metrics")
                metrics = {
                    'Expected Annual Return (%)': f"{exp_return * 100:.2f}",
                    'Annual Volatility (%)': f"{vol * 100:.2f}",
                    'Sharpe Ratio': f"{sharpe:.2f}"
                }
                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                st.table(metrics_df)
                
                # Visualize Portfolio Allocation
                st.subheader("📊 Portfolio Allocation Pie Chart")
                fig_pie = px.pie(
                    names=weights_df.index,
                    values=weights_df['Weight'],
                    title="Portfolio Allocation",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Plot Cumulative Returns
                st.subheader("📈 Cumulative Returns")
                cumulative_returns = (1 + returns.dot(pd.Series(weights))).cumprod()
                fig_cum = px.line(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    labels={'x': 'Date', 'y': 'Cumulative Returns'},
                    title="Portfolio Cumulative Returns"
                )
                st.plotly_chart(fig_cum, use_container_width=True)
                
                # Efficient Frontier Plot
                st.subheader("📈 Efficient Frontier")
                mu = expected_returns.mean_historical_return(adj_close)
                S = risk_models.sample_cov(adj_close)
                ef_plot = EfficientFrontier(mu, S)
                fig_ef_image = plot_efficient_frontier(mu, S)
                st.image(fig_ef_image, caption='Efficient Frontier', use_column_width=True)
                
                # Discrete Allocation
                st.subheader("💼 Discrete Allocation")
                latest_prices = get_latest_prices(adj_close)
                total_portfolio_value = st.number_input(
                    "💵 Enter Total Portfolio Value (USD)",
                    min_value=0.0,
                    value=10000.0,
                    step=100.0
                )
                discrete_alloc = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
                allocation, leftover = discrete_alloc.lp_portfolio()
                
                st.write("**Discrete Allocation (Number of Shares to Buy):**")
                allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
                allocation_df = allocation_df[allocation_df['Shares'] > 0]
                st.table(allocation_df)
                st.write(f"**Leftover Funds:** ${leftover:.2f}")
                
                # Download discrete allocation
                st.download_button(
                    label="💾 Download Discrete Allocation as CSV",
                    data=allocation_df.to_csv().encode('utf-8'),
                    file_name="discrete_allocation.csv",
                    mime="text/csv"
                )
    
    if __name__ == "__main__":
        main()
