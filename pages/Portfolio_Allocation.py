#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Optimization Dashboard - Streamlit Version

Author: Thodoreskourtales
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid
from pypfopt import black_litterman, risk_models, expected_returns
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
import seaborn as sns
import requests
import scipy.cluster.hierarchy as sch
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf
import os

# Suppress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the plotting style
sns.set(style="white")

# Streamlit page configuration
st.set_page_config(
    page_title="📈 Portfolio Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# New Function to Validate Tickers
def validate_tickers(tickers):
    """
    Validate the provided tickers by checking if they have valid data over the past month.
    """
    tickers = list(set(ticker.strip().upper() for ticker in tickers))  # Normalize to uppercase
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=30)

    valid_tickers = []
    invalid_tickers = []
    for ticker in tickers:
        try:
            # Download data for the past month to check validity
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception as e:
            st.warning(f"Failed to validate ticker {ticker}: {e}")
            invalid_tickers.append(ticker)

    if invalid_tickers:
        st.warning(f"⚠️ The following tickers are invalid or have insufficient data and will be excluded: {', '.join(invalid_tickers)}")

    return valid_tickers

# New Function to Fetch Data
def fetch_data(tickers, start_date, end_date):
    """
    Fetch adjusted close prices and market capitalizations for given tickers from Yahoo Finance.
    """
    tickers = [ticker.strip().upper() for ticker in tickers]  # Normalize to uppercase
    st.write("Fetching data for:", tickers)
    
    try:
        # Fetch historical data
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if len(tickers) == 1:
            # If only one ticker, yfinance returns a DataFrame with columns like 'Open', 'High', etc.
            adj_close = data['Adj Close'].to_frame()
            adj_close.columns = tickers
        else:
            if 'Adj Close' in data.columns:
                adj_close = data['Adj Close']
            elif 'Close' in data.columns:
                adj_close = data['Close']
            else:
                raise KeyError("No 'Adj Close' or 'Close' column in downloaded data.")
    except Exception as e:
        st.error(f"Error fetching data for tickers: {e}")
        return pd.DataFrame(), {}

    # Validate which tickers were successfully fetched
    valid_tickers = adj_close.columns.tolist()
    invalid_tickers = list(set(tickers) - set(valid_tickers))
    
    if invalid_tickers:
        st.warning(f"⚠️ The following tickers could not be fetched: {', '.join(invalid_tickers)}")

    # Fetch market capitalizations
    mcaps = {}
    for ticker in valid_tickers:
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap')
            if market_cap:
                mcaps[ticker] = market_cap
            else:
                st.warning(f"⚠️ Market capitalization for {ticker} not found.")
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch market capitalization for {ticker}: {e}")

    return adj_close, mcaps

# Function to calculate daily returns
def calculate_daily_returns(data):
    """Calculate daily returns of the given data."""
    return data.pct_change().ffill().dropna()

# Function to calculate portfolio performance
def calculate_portfolio_performance(weights, returns, risk_free_rate=0.0):
    """
    Calculate the expected annual return, annual volatility, and Sharpe ratio for a portfolio.
    """
    portfolio_return = np.dot(weights, returns.mean()) * 252  # Assuming 252 trading days in a year
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualize volatility
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Function for Mean-Variance Optimization
def mean_variance_optimization(returns, risk_free_rate=0.0):
    """Optimize portfolio using mean-variance optimization."""
    num_assets = returns.shape[1]
    expected_returns = returns.mean()
    cov_matrix = returns.cov()

    def portfolio_return(weights):
        return np.dot(weights, expected_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def sharpe_ratio(weights):
        return (portfolio_return(weights) - risk_free_rate) / portfolio_volatility(weights)

    def objective_function(weights):
        return -sharpe_ratio(weights)

    initial_guess = num_assets * [1. / num_assets]
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    optimized = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if not optimized.success:
        st.error("❌ Mean-Variance Optimization failed.")
        return None

    return dict(zip(returns.columns, optimized.x))

# Function for Black-Litterman Allocation
def black_litterman_allocation(market_prices, mcaps, cov_matrix, viewdict, tau=0.05):
    """
    Optimize portfolio using Black-Litterman model.

    Parameters:
    - market_prices (pd.DataFrame): Historical adjusted close prices (DataFrame).
    - mcaps (dict): Market capitalizations for each ticker.
    - cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
    - viewdict (dict): User-specified views on expected returns.
    - tau (float): Scaling factor for the covariance matrix.

    Returns:
    - cleaned_weights (dict): Optimized portfolio weights.
    - performance (tuple): Portfolio performance (expected return, volatility, Sharpe ratio).
    """
    try:
        # Ensure market_prices is converted to a pd.Series (latest prices)
        latest_prices = market_prices.iloc[-1]
        if not isinstance(latest_prices, pd.Series):
            raise ValueError("market_prices must be convertible to a pd.Series of latest prices.")

        # Ensure market capitalizations are complete
        if any(pd.isna(mcaps.get(ticker)) for ticker in latest_prices.index):
            raise ValueError("Market capitalizations are missing for some tickers.")

        # Compute implied risk aversion
        delta = black_litterman.market_implied_risk_aversion(latest_prices)

        # Compute market-implied prior returns
        prior = black_litterman.market_implied_prior_returns(mcaps, delta, cov_matrix)

        # Construct Black-Litterman model
        bl = BlackLittermanModel(cov_matrix, pi=prior, absolute_views=viewdict, tau=tau)
        posterior_rets = bl.bl_returns()
        posterior_cov = bl.bl_cov()

        # Solve for optimal weights
        ef = EfficientFrontier(posterior_rets, posterior_cov)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)

        return cleaned_weights, performance

    except Exception as e:
        st.error(f"❌ Black-Litterman Optimization failed: {e}")
        return None, None

# Function for Risk Parity Optimization
def risk_parity_optimization(returns):
    """Optimize portfolio using risk parity method."""
    num_assets = returns.shape[1]
    cov_matrix = returns.cov()

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def risk_contribution(weights):
        total_volatility = portfolio_volatility(weights)
        marginal_risk_contribution = np.dot(cov_matrix, weights)
        risk_contribution = weights * marginal_risk_contribution / total_volatility
        return risk_contribution

    def objective_function(weights):
        risk_contributions = risk_contribution(weights)
        target_risk = np.ones_like(weights) / num_assets
        return np.sum((risk_contributions - target_risk) ** 2)

    initial_guess = num_assets * [1. / num_assets]
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    optimized = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if not optimized.success:
        st.error("❌ Risk Parity Optimization failed.")
        return None

    return dict(zip(returns.columns, optimized.x))

# Function for Mean-CVaR Optimization
def mean_cvar_optimization(returns, confidence_level=0.95):
    """Optimize portfolio using mean-CVaR method."""
    num_assets = returns.shape[1]
    expected_returns = returns.mean().values.reshape(-1, 1)

    weights = cp.Variable(num_assets)
    portfolio_return = expected_returns.T @ weights
    portfolio_risk = cp.norm(returns.values @ weights, 2)

    alpha = cp.Parameter(nonneg=True)
    alpha.value = confidence_level
    z = cp.Variable(returns.shape[0])
    t = cp.Variable()

    objective = cp.Minimize(t + (1 / ((1 - alpha.value) * returns.shape[0])) * cp.sum(z))
    constraints = [z >= 0, z >= -returns.values @ weights - t]
    constraints += [cp.sum(weights) == 1, weights >= 0]

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        return dict(zip(returns.columns, weights.value))
    except Exception as e:
        st.error(f"❌ Mean-CVaR Optimization failed: {e}")
        return None

# Function to perform Hierarchical Risk Parity Optimization
def hierarchical_risk_parity_optimization(returns, linkage_method='single'):
    """Optimize portfolio using hierarchical risk parity (HRP) method."""
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()

    # Ensure covariance matrix does not contain NaN values
    if cov_matrix.isnull().values.any():
        st.error("❌ Covariance matrix contains NaN values. HRP optimization cannot proceed.")
        return None

    # Compute the linkage matrix for hierarchical clustering
    dist = sch.distance.pdist(corr_matrix, metric='euclidean')
    linkage_mat = sch.linkage(dist, method=linkage_method)

    # Form the clusters and get sorted indices
    cluster_tree = sch.to_tree(linkage_mat, rd=False)
    sorted_indices = get_quasi_diag(cluster_tree)
    sorted_returns = returns.iloc[:, sorted_indices]

    # Allocate weights based on inverse variance
    weights = hrp_weights(cov_matrix, sorted_indices)

    # Check if the weights contain NaN values
    if pd.isnull(weights).any():
        st.error("❌ HRP optimization resulted in NaN values for weights.")
        return None

    return dict(zip(sorted_returns.columns, weights))

def get_quasi_diag(link):
    """Get the order of items from hierarchical clustering."""
    if link.is_leaf():
        return [link.id]
    else:
        return get_quasi_diag(link.get_left()) + get_quasi_diag(link.get_right())

def hrp_weights(cov_matrix, sorted_indices):
    """Compute hierarchical risk parity weights."""
    weights = pd.Series(1.0, index=sorted_indices)
    cluster_items = [sorted_indices]

    # Iterate over clusters and allocate weights
    while len(cluster_items) > 0:
        new_clusters = []
        for cluster in cluster_items:
            if len(cluster) > 1:
                # Split the cluster into two halves
                split_point = len(cluster) // 2
                left_cluster = cluster[:split_point]
                right_cluster = cluster[split_point:]

                # Calculate the average covariance within each cluster
                left_cov = cov_matrix.loc[left_cluster, left_cluster].values.mean()
                right_cov = cov_matrix.loc[right_cluster, right_cluster].values.mean()

                # Allocate weights based on inverse variance
                if (left_cov + right_cov) == 0:
                    alloc_factor = 0.5
                else:
                    alloc_factor = 1 - left_cov / (left_cov + right_cov)
                weights[left_cluster] *= alloc_factor
                weights[right_cluster] *= (1 - alloc_factor)

                # Add new clusters to the list
                new_clusters.append(left_cluster)
                new_clusters.append(right_cluster)

        cluster_items = new_clusters

    # Normalize the weights to ensure they sum to 1
    return weights / weights.sum()

# Function to plot optimized portfolios
def plot_optimized_portfolios(optimized_portfolios, data, start_date, end_date):
    """Plot cumulative returns for optimized portfolios and individual assets."""
    fig = make_subplots()

    symbols = data.columns
    stock_returns = calculate_daily_returns(data)
    
    # Plot individual stocks
    for symbol in symbols:
        cumulative_stock_returns = (1 + stock_returns[symbol]).cumprod() - 1
        fig.add_trace(
            px.line(x=cumulative_stock_returns.index, y=cumulative_stock_returns, name=symbol).data[0]
        )

    # Plot optimized portfolios
    for method, allocations in optimized_portfolios.items():
        weights = np.array(list(allocations.values()))
        portfolio_returns = (stock_returns * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        fig.add_trace(
            px.line(x=cumulative_returns.index, y=cumulative_returns, name=f'Optimized Portfolio ({method})', line=dict(dash='dash')).data[0]
        )

    fig.update_layout(
        title='📈 Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        legend_title='Legend',
        template='plotly_white',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# Function to plot asset allocation as a pie chart
def plot_asset_allocation(weights, title):
    """Plot the asset allocation as a pie chart."""
    labels = list(weights.keys())
    sizes = list(weights.values())
    colors = px.colors.qualitative.Plotly

    fig = px.pie(names=labels, values=sizes, title=title, color=labels, color_discrete_sequence=colors)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, template='plotly_white')

    st.plotly_chart(fig, use_container_width=True)

# Function to calculate weighted average allocation
def calculate_weighted_average_allocation(optimized_portfolios):
    """Calculate the weighted average allocation based on user-defined weights for each optimization technique."""
    techniques = list(optimized_portfolios.keys())
    weights = {}
    st.subheader("📊 Weighted Average Allocation")

    # User inputs for weights
    st.markdown("Assign weights to each optimization technique:")
    cols = st.columns(len(techniques))
    for i, technique in enumerate(techniques):
        weights[technique] = cols[i].number_input(
            f"Weight for {technique}",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key=f"weight_{technique}"
        )

    total_weight = sum(weights.values())
    if total_weight == 0:
        st.error("❌ Total weight cannot be zero. Please assign positive weights.")
        return

    # Normalize weights
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate average allocation
    stocks = optimized_portfolios[techniques[0]].keys()
    avg_allocation = {stock: 0 for stock in stocks}

    for technique, weight in normalized_weights.items():
        for stock, allocation in optimized_portfolios[technique].items():
            avg_allocation[stock] += allocation * weight

    # User input for total USD amount
    total_usd = st.number_input(
        "💵 Enter the total USD amount to allocate:",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )

    usd_allocation = {stock: allocation * total_usd for stock, allocation in avg_allocation.items()}

    # Display average allocation
    st.markdown("### 💼 Average Allocation for Each Stock:")
    allocation_df = pd.DataFrame({
        'Stock': avg_allocation.keys(),
        'Allocation (%)': [f"{v*100:.2f}%" for v in avg_allocation.values()],
        'Allocation (USD)': [f"${v:.2f}" for v in usd_allocation.values()]
    })
    st.table(allocation_df)

    # Plot average allocation
    plot_asset_allocation(avg_allocation, "📊 Average Asset Allocation")

# Function to perform grid search optimization
def grid_search_optimization(returns, risk_free_rate, market_prices, mcaps, cov_matrix, viewdict):
    """Perform grid search to find the best parameters for portfolio optimization."""
    best_params = {}
    best_sharpe = -np.inf
    
    param_grid = {
        'tau': [0.01, 0.025, 0.05],
        'confidence_level': [0.9, 0.95, 0.99]
    }
    
    for params in ParameterGrid(param_grid):
        try:
            bl_weights, _ = black_litterman_allocation(market_prices, mcaps, cov_matrix, viewdict, tau=params['tau'])
            if bl_weights is None:
                continue
            mv_weights = mean_variance_optimization(returns, risk_free_rate)
            rp_weights = risk_parity_optimization(returns)
            cvar_weights = mean_cvar_optimization(returns, confidence_level=params['confidence_level'])
            
            def calculate_sharpe(weights):
                portfolio_return = np.dot(weights, returns.mean())
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
                return (portfolio_return - risk_free_rate) / portfolio_volatility

            portfolios = {
                'Black-Litterman': bl_weights,
                'Mean-Variance': mv_weights,
                'Risk-Parity': rp_weights,
                'Mean-CVaR': cvar_weights
            }

            for method, weights in portfolios.items():
                if weights is None:
                    continue
                sharpe = calculate_sharpe(np.array(list(weights.values())))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'method': method, 'weights': weights, 'params': params}
        except Exception as e:
            st.error(f"❌ Error with parameters {params}: {e}")
            continue

    return best_params

# Main Streamlit App
def main():
    st.title("📈 Portfolio Optimization Dashboard")
    st.markdown("""
    This application fetches and analyzes financial data for specified companies. 
    Enter the ticker symbols, select optimization techniques, and explore various 
    financial metrics and portfolio allocations.
    """)

    # Sidebar Inputs
    st.sidebar.header("🔧 Input Parameters")
    
    tickers_input = st.sidebar.text_input(
        "📊 Enter Ticker Symbols (comma-separated)",
        value="AAPL, MSFT"
    )
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    
    start_date = st.sidebar.date_input(
        "📅 Start Date",
        value=datetime.date.today() - datetime.timedelta(days=365),
        min_value=datetime.date(1900, 1, 1),
        max_value=datetime.date.today()
    )
    
    end_date = st.sidebar.date_input(
        "📅 End Date",
        value=datetime.date.today(),
        min_value=start_date,
        max_value=datetime.date.today()
    )
    
    # Set Default Risk-Free Rate (e.g., 2%)
    risk_free_rate = st.sidebar.number_input(
        "💵 Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Assumed risk-free rate for Sharpe Ratio calculations."
    ) / 100  # Convert to decimal
    
    # Selection of Optimization Techniques
    st.sidebar.header("📈 Optimization Techniques")
    techniques_options = {
        "Mean Variance Optimization": "Mean Variance Optimization",
        "Black-Litterman Model": "Black-Litterman Model",
        "Risk Parity Optimization": "Risk Parity Optimization",
        "Mean-CVaR Optimization": "Mean-CVaR Optimization",
        "Hierarchical Risk Parity": "Hierarchical Risk Parity"
    }
    selected_techniques = st.sidebar.multiselect(
        "🔍 Select Optimization Techniques:",
        options=list(techniques_options.values()),
        default=list(techniques_options.values())  # Select all by default
    )
    
    # Run Analysis Button
    run_analysis = st.sidebar.button("🚀 Run Analysis")

    if run_analysis:
        if not tickers:
            st.error("❌ Please enter at least one ticker symbol.")
            return

        # Validate Tickers
        with st.spinner("🔍 Validating ticker symbols..."):
            valid_tickers = validate_tickers(tickers)
            if not valid_tickers:
                st.error("❌ No valid tickers available after validation.")
                return
            else:
                st.success(f"✅ Valid tickers: {', '.join(valid_tickers)}")

        # Fetch Market Data
        with st.spinner("📥 Fetching market data..."):
            market_prices, mcaps = fetch_data(valid_tickers, start_date, end_date)
            if market_prices.empty:
                st.error("❌ Failed to fetch market data. Please check your ticker symbols and internet connection.")
                return
            else:
                st.success("✅ Market data fetched successfully.")

        # Handle Single Ticker Scenario
        if isinstance(market_prices, pd.Series):
            market_prices = market_prices.to_frame()

        # Calculate Returns
        returns = calculate_daily_returns(market_prices)
        if returns.empty:
            st.error("❌ No return data available to perform optimizations.")
            return

        # Get user views for Black-Litterman (if selected)
        viewdict = {}
        if "Black-Litterman Model" in selected_techniques:
            st.subheader("📝 Enter Your Views for Black-Litterman Model")
            st.markdown("Provide your expected returns (in percentage) for each ticker:")
            for ticker in valid_tickers:
                view = st.number_input(
                    f"Expected Return for {ticker} (%)",
                    value=0.0,
                    key=f"view_{ticker}"
                )
                viewdict[ticker] = view / 100  # Convert to decimal

        # Initialize dictionaries to store optimized portfolios and performance metrics
        optimized_portfolios = {}
        performance_metrics = {}
        
        # Perform selected optimization techniques
        for technique in selected_techniques:
            st.markdown(f"### 🔍 {technique}")
            with st.spinner(f"🛠️ Performing {technique}..."):
                if technique == "Mean Variance Optimization":
                    mv_weights = mean_variance_optimization(returns, risk_free_rate)
                    if mv_weights:
                        optimized_portfolios['Mean Variance'] = mv_weights
                        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_performance(
                            np.array(list(mv_weights.values())),
                            returns,
                            risk_free_rate
                        )
                        performance_metrics['Mean Variance'] = {
                            'Expected Return (%)': f"{portfolio_return*100:.2f}",
                            'Volatility (%)': f"{portfolio_volatility*100:.2f}",
                            'Sharpe Ratio': f"{sharpe_ratio:.2f}"
                        }
                        st.success("✅ Mean Variance Optimization completed successfully.")
                        plot_asset_allocation(mv_weights, "📊 Mean Variance Asset Allocation")
                elif technique == "Black-Litterman Model":
                    if not viewdict:
                        st.warning("⚠️ No views provided. Skipping Black-Litterman Optimization.")
                        continue
                    if not mcaps:
                        st.warning("⚠️ Market capitalizations not available. Skipping Black-Litterman Optimization.")
                        continue
                    bl_weights, bl_performance = black_litterman_allocation(market_prices, mcaps, returns.cov(), viewdict, tau=0.05)
                    if bl_weights:
                        optimized_portfolios['Black-Litterman'] = bl_weights
                        performance_metrics['Black-Litterman'] = {
                            'Expected Return (%)': f"{bl_performance[0]*100:.2f}",
                            'Volatility (%)': f"{bl_performance[1]*100:.2f}",
                            'Sharpe Ratio': f"{bl_performance[2]:.2f}"
                        }
                        st.success("✅ Black-Litterman Optimization completed successfully.")
                        plot_asset_allocation(bl_weights, "📊 Black-Litterman Asset Allocation")
                elif technique == "Risk Parity Optimization":
                    rp_weights = risk_parity_optimization(returns)
                    if rp_weights:
                        optimized_portfolios['Risk Parity'] = rp_weights
                        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_performance(
                            np.array(list(rp_weights.values())),
                            returns,
                            risk_free_rate
                        )
                        performance_metrics['Risk Parity'] = {
                            'Expected Return (%)': f"{portfolio_return*100:.2f}",
                            'Volatility (%)': f"{portfolio_volatility*100:.2f}",
                            'Sharpe Ratio': f"{sharpe_ratio:.2f}"
                        }
                        st.success("✅ Risk Parity Optimization completed successfully.")
                        plot_asset_allocation(rp_weights, "📊 Risk Parity Asset Allocation")
                elif technique == "Mean-CVaR Optimization":
                    cvar_weights = mean_cvar_optimization(returns, confidence_level=0.95)
                    if cvar_weights:
                        optimized_portfolios['Mean CVaR'] = cvar_weights
                        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_performance(
                            np.array(list(cvar_weights.values())),
                            returns,
                            risk_free_rate
                        )
                        performance_metrics['Mean CVaR'] = {
                            'Expected Return (%)': f"{portfolio_return*100:.2f}",
                            'Volatility (%)': f"{portfolio_volatility*100:.2f}",
                            'Sharpe Ratio': f"{sharpe_ratio:.2f}"
                        }
                        st.success("✅ Mean-CVaR Optimization completed successfully.")
                        plot_asset_allocation(cvar_weights, "📊 Mean-CVaR Asset Allocation")
                elif technique == "Hierarchical Risk Parity":
                    hrp_weights_dict = hierarchical_risk_parity_optimization(returns, linkage_method='single')
                    if hrp_weights_dict:
                        optimized_portfolios['Hierarchical Risk Parity'] = hrp_weights_dict
                        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_performance(
                            np.array(list(hrp_weights_dict.values())),
                            returns,
                            risk_free_rate
                        )
                        performance_metrics['Hierarchical Risk Parity'] = {
                            'Expected Return (%)': f"{portfolio_return*100:.2f}",
                            'Volatility (%)': f"{portfolio_volatility*100:.2f}",
                            'Sharpe Ratio': f"{sharpe_ratio:.2f}"
                        }
                        st.success("✅ Hierarchical Risk Parity Optimization completed successfully.")
                        plot_asset_allocation(hrp_weights_dict, "📊 Hierarchical Risk Parity Asset Allocation")

        if not optimized_portfolios:
            st.error("❌ No portfolios were successfully optimized.")
            return

        # Plot optimized portfolios cumulative returns
        st.subheader("📈 Optimized Portfolios Cumulative Returns")
        plot_optimized_portfolios(optimized_portfolios, market_prices, start_date, end_date)

        # Display performance metrics
        st.subheader("📊 Portfolio Performance Metrics")
        metrics_df = pd.DataFrame(performance_metrics).T
        st.table(metrics_df)

        # Calculate Weighted Average Allocation
        st.subheader("📊 Weighted Average Allocation")
        calculate_weighted_average_allocation(optimized_portfolios)

        # Grid Search Optimization (Optional)
        st.subheader("🔍 Grid Search Optimization")
        with st.spinner("🕵️ Performing grid search optimization..."):
            best_params = grid_search_optimization(returns, risk_free_rate, market_prices, mcaps, returns.cov(), viewdict)
            if best_params:
                st.markdown(f"**Best Parameters Found:**")
                st.json(best_params)
            else:
                st.warning("⚠️ Grid search did not find better parameters.")

        # Note: The `console_output` section has been removed as it wasn't utilized in the updated functions.

if __name__ == "__main__":
    main()
