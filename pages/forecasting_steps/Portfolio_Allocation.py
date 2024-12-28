#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:21:58 2024

Author: Theodoros Kourtales
"""

# pages/forecasting_steps/3_Portfolio_Allocation.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import risk_models, expected_returns, EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import plotting
import matplotlib.pyplot as plt
import io

def main():
    st.header("📈 Step 3: Portfolio Allocation")
    
    # Check if cleaned data is available
    if 'cleaned_data' not in st.session_state or 'symbol' not in st.session_state:
        st.warning("No cleaned data found. Please complete Steps 1 and 2 first.")
        st.markdown("### ➡️ Return to the Previous Steps:")
        if st.button("🔙 Go to Step 1: Fetch Raw Data"):
            st.session_state["current_page"] = "1_Fetch_Raw_Data"
            st.experimental_rerun()
        return
    
    # Retrieve cleaned data and symbols
    cleaned_data = st.session_state['cleaned_data']
    symbols = st.session_state['symbol']
    
    st.write("**Cleaned Data Preview:**")
    st.dataframe(cleaned_data.head())
    
    # Ensure 'ds' is datetime and 'y' is numeric
    if not (pd.api.types.is_datetime64_any_dtype(cleaned_data['ds']) and pd.api.types.is_numeric_dtype(cleaned_data['y'])):
        st.error("Data types are incorrect. 'ds' should be datetime and 'y' should be numeric.")
        return
    
    # Pivot data for multiple tickers
    st.subheader("📊 Calculate Daily Returns")
    try:
        # If multiple tickers, 'symbol' column should exist
        if 'symbol' in cleaned_data.columns:
            returns = cleaned_data.pivot(index='ds', columns='symbol', values='y').pct_change().dropna()
        else:
            # Single ticker scenario
            returns = cleaned_data.set_index('ds')['y'].pct_change().to_frame()
            returns.columns = [symbols[0]]
            returns.dropna(inplace=True)
    except Exception as e:
        st.error(f"Error pivoting data: {e}")
        return
    
    if returns.empty:
        st.error("No returns data available after processing.")
        return
    
    st.write("**Daily Returns Preview:**")
    st.dataframe(returns.head())
    
    # Display correlation matrix if multiple tickers
    if len(symbols) > 1:
        st.subheader("🔗 Correlation Matrix")
        corr_matrix = returns.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix of Returns")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Portfolio Optimization
    st.subheader("🛠️ Portfolio Optimization Techniques")
    
    # Expected Returns and Covariance Matrix
    mu = expected_returns.mean_historical_return(cleaned_data.pivot(index='ds', columns='symbol', values='y'))
    S = risk_models.sample_cov(cleaned_data.pivot(index='ds', columns='symbol', values='y'))
    
    # Efficient Frontier
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    
    st.write("**Optimized Portfolio Weights:**")
    weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
    weights_df = weights_df[weights_df['Weight'] > 0]
    st.dataframe(weights_df)
    
    st.write(f"**Expected Annual Return:** {expected_annual_return*100:.2f}%")
    st.write(f"**Annual Volatility:** {annual_volatility*100:.2f}%")
    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
    
    # Plot Efficient Frontier
    st.subheader("📈 Efficient Frontier")
    fig_ef = plt.figure(figsize=(10,6))
    plotting.plot_efficient_frontier(ef, show_assets=True)
    plt.tight_layout()
    
    # Convert Matplotlib figure to Plotly
    buf = io.BytesIO()
    fig_ef.savefig(buf, format="png")
    buf.seek(0)
    image = buf.read()
    fig_plotly = px.imshow(np.array(plt.imread(buf)), binary_string=True)
    st.image(image, caption='Efficient Frontier', use_column_width=True)
    plt.close(fig_ef)
    
    # Display Pie Chart of Portfolio Allocation
    st.subheader("📊 Portfolio Allocation Pie Chart")
    fig_pie = px.pie(
        names=weights_df.index,
        values=weights_df['Weight'],
        title="Portfolio Allocation",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Optional: Discrete Allocation (number of shares to buy)
    st.subheader("💼 Discrete Allocation")
    latest_prices = get_latest_prices(cleaned_data.pivot(index='ds', columns='symbol', values='y'))
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    discrete_alloc = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
    allocation, leftover = discrete_alloc.lp_portfolio()
    
    st.write("**Discrete Allocation:**")
    allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
    st.table(allocation_df)
    st.write(f"**Leftover Funds:** ${leftover:.2f}")

if __name__ == "__main__":
    main()