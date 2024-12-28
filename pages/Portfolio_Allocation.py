#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Optimization Dashboard - Streamlit Version

Author: Thodoreskourtales
"""

import streamlit as st
import subprocess
import sys
import importlib
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
    page_title="Portfolio Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to fetch data from URL with caching
@st.cache_data(show_spinner=False)
def fetch_data(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err} for URL: {url}")
    except Exception as err:
        st.error(f"An error occurred: {err} for URL: {url}")
    return None

# Function to compute CAGR
def calculate_cagr(start_value, end_value, periods):
    try:
        return ((end_value / start_value) ** (1 / periods) - 1) * 100
    except ZeroDivisionError:
        return None

# Function to fetch and process data for a specific ticker
@st.cache_data(show_spinner=False)
def get_company_data(ticker, headers):
    company_data = fetch_data("https://www.sec.gov/files/company_tickers.json", headers)
    
    if not company_data:
        st.error("Failed to retrieve company tickers.")
        return None, None

    # Find the CIK for the given ticker
    cik = None
    for key, value in company_data.items():
        if value['ticker'].upper() == ticker.upper():
            cik = value['cik_str']
            break

    if cik is None:
        st.error(f"{ticker} not found in the SEC company tickers list.")
        return None, None

    # Add leading zeros to CIK
    cik = str(cik).zfill(10)

    # Get company specific filing metadata
    filing_metadata = fetch_data(f'https://data.sec.gov/submissions/CIK{cik}.json', headers)
    if not filing_metadata:
        st.error("Failed to retrieve filing metadata.")
        return None, None

    # Parse the metadata
    filings = filing_metadata.get('filings', {}).get('recent', {})
    all_forms = pd.DataFrame.from_dict(filings)
    
    # Get company facts data
    company_facts = fetch_data(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json', headers)
    if not company_facts:
        st.error("Failed to retrieve company facts data.")
        return None, None

    return all_forms, company_facts

# Function to calculate financial ratios
def calculate_ratios(company_facts, ticker):
    ratios = {}
    us_gaap = company_facts.get('facts', {}).get('us-gaap', {})

    def get_value(key):
        try:
            # Attempt to get the latest value
            values = us_gaap.get(key, {}).get('units', {}).get('USD', [])
            if values:
                # Return the most recent value
                return values[-1].get('val')
            return None
        except Exception:
            return None

    # Fetch necessary financial metrics
    current_assets = get_value('AssetsCurrent')
    current_liabilities = get_value('LiabilitiesCurrent')
    total_assets = get_value('Assets')
    total_liabilities = get_value('Liabilities')
    cash_and_cash_equivalents = get_value('CashAndCashEquivalentsAtCarryingValue')
    inventory = get_value('InventoryNet')
    net_income = get_value('NetIncomeLoss')
    shareholder_equity = get_value('StockholdersEquity')
    net_sales = get_value('SalesRevenueNet')
    gross_profit = get_value('GrossProfit')
    operating_income = get_value('OperatingIncomeLoss')
    interest_expense = get_value('InterestExpense')
    ebit = get_value('EarningsBeforeInterestAndTaxes')
    ebitda = get_value('EarningsBeforeInterestTaxesDepreciationAndAmortization')
    operating_cash_flow = get_value('NetCashProvidedByUsedInOperatingActivities')
    total_debt_service = get_value('InterestAndDebtExpense')

    # Fetch real market data using yfinance
    try:
        stock = yf.Ticker(ticker)
        market_price_per_share = stock.info.get('currentPrice', 100)  # Default to 100 if not available
        earnings_per_share = stock.info.get('trailingEps', 1)  # Default to 1 if not available
        book_value_per_share = stock.info.get('bookValue', 50)  # Default to 50 if not available
        annual_dividends_per_share = stock.info.get('dividendRate', 2)  # Default to 2 if not available
    except Exception as e:
        st.warning(f"Failed to fetch market data for {ticker}: {e}")
        market_price_per_share = 100
        earnings_per_share = 1
        book_value_per_share = 50
        annual_dividends_per_share = 2

    try:
        # Liquidity Ratios
        ratios['Current Ratio'] = current_assets / current_liabilities if current_assets and current_liabilities else None
        ratios['Quick Ratio'] = (current_assets - inventory) / current_liabilities if current_assets and inventory and current_liabilities else None
        ratios['Cash Ratio'] = cash_and_cash_equivalents / current_liabilities if cash_and_cash_equivalents and current_liabilities else None
        ratios['Operating Cash Flow Ratio'] = operating_cash_flow / current_liabilities if operating_cash_flow and current_liabilities else None
        ratios['Working Capital'] = current_assets - current_liabilities if current_assets and current_liabilities else None

        # Profitability Ratios
        ratios['ROA'] = (net_income / total_assets) * 100 if net_income and total_assets else None
        ratios['ROE'] = (net_income / shareholder_equity) * 100 if net_income and shareholder_equity else None
        ratios['Gross Margin'] = (gross_profit / net_sales) * 100 if gross_profit and net_sales else None
        ratios['Operating Margin'] = (operating_income / net_sales) * 100 if operating_income and net_sales else None
        ratios['Net Profit Margin'] = (net_income / net_sales) * 100 if net_income and net_sales else None
        ratios['ROI'] = (net_income / total_assets) * 100 if net_income and total_assets else None
        ratios['EBIT Margin'] = (ebit / net_sales) * 100 if ebit and net_sales else None
        ratios['EBITDA Margin'] = (ebitda / net_sales) * 100 if ebitda and net_sales else None

        # Leverage Ratios
        ratios['Debt-to-Equity Ratio'] = total_liabilities / shareholder_equity if total_liabilities and shareholder_equity else None
        ratios['Debt Ratio'] = total_liabilities / total_assets if total_liabilities and total_assets else None
        ratios['Interest Coverage Ratio'] = operating_income / interest_expense if operating_income and interest_expense else None
        ratios['Equity Ratio'] = shareholder_equity / total_assets if shareholder_equity and total_assets else None
        ratios['Debt Service Coverage Ratio (DSCR)'] = ebit / total_debt_service if ebit and total_debt_service else None

        # Efficiency Ratios
        ratios['Asset Turnover Ratio'] = net_sales / total_assets if net_sales and total_assets else None
        ratios['Inventory Turnover Ratio'] = net_sales / inventory if net_sales and inventory else None
        ratios['Receivables Turnover Ratio'] = net_sales / inventory if net_sales and inventory else None  # Possible typo: should be Receivables

        # Valuation Ratios
        ratios['P/E Ratio'] = market_price_per_share / earnings_per_share if market_price_per_share and earnings_per_share else None
        ratios['P/B Ratio'] = market_price_per_share / book_value_per_share if market_price_per_share and book_value_per_share else None
        ratios['Dividend Yield'] = (annual_dividends_per_share / market_price_per_share) * 100 if annual_dividends_per_share and market_price_per_share else None

    except ZeroDivisionError as e:
        st.error(f"Error calculating ratios for {ticker}: {e}")
        return None

    return ratios

# Function to log and plot financial ratios
def log_and_plot_ratios(ratios, company_name, save_plots=False, plot_save_path=None, console_output=None):
    # Categorize ratios
    liquidity_ratios = {k: v for k, v in ratios.items() if ('Ratio' in k and 'Current' in k) or k in ['Operating Cash Flow Ratio', 'Working Capital']}
    profitability_ratios = {k: v for k, v in ratios.items() if 'Margin' in k or k in ['ROA', 'ROE', 'ROI', 'EBITDA Margin']}
    leverage_ratios = {k: v for k, v in ratios.items() if 'Debt' in k or 'Interest' in k or 'Equity' in k or 'DSCR' in k}
    efficiency_ratios = {k: v for k, v in ratios.items() if 'Turnover' in k}
    valuation_ratios = {k: v for k, v in ratios.items() if 'P/E' in k or 'P/B' in k or 'Dividend Yield' in k}

    categories = {
        'Liquidity Ratios': liquidity_ratios,
        'Profitability Ratios': profitability_ratios,
        'Leverage Ratios': leverage_ratios,
        'Efficiency Ratios': efficiency_ratios,
        'Valuation Ratios': valuation_ratios,
    }

    if save_plots and plot_save_path:
        os.makedirs(plot_save_path, exist_ok=True)

    # Plotting using Plotly for interactivity
    for category, data in categories.items():
        df = pd.DataFrame(list(data.items()), columns=['Ratio', 'Value']).dropna()
        if not df.empty:
            fig = px.bar(df, x='Ratio', y='Value',
                         title=f"{company_name} - {category}",
                         labels={'Value': 'Percentage (%)'},
                         text='Value')
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                              xaxis_tickangle=-45, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            # Save plot if required
            if save_plots and plot_save_path:
                fig.write_image(os.path.join(plot_save_path, f"{company_name}_{category.replace(' ', '_')}.png"))

    # Display ratios in tables
    for category, data in categories.items():
        df = pd.DataFrame(list(data.items()), columns=['Ratio', 'Value']).dropna()
        if not df.empty:
            st.subheader(category)
            st.table(df.style.format({'Value': '{:.2f}'}))

    # Log ratios
    for category, data in categories.items():
        for ratio, value in data.items():
            message = f"{company_name} - {ratio}: {value:.2f}%" if value is not None else f"{company_name} - {ratio}: N/A"
            console_output.append(message)

# Function to calculate daily returns
def calculate_daily_returns(data):
    """Calculate daily returns of the given data."""
    return data.pct_change().ffill().dropna()

# Function to fetch Treasury data from Alpha Vantage
def fetch_treasury_data(api_url, api_key):
    """Fetch Treasury data from the provided API URL using the given API key."""
    try:
        response = requests.get(f"{api_url}&apikey={api_key}")
        response.raise_for_status()
        data = response.json()
        # Assuming the data has a 'data' key with relevant information
        return pd.DataFrame(data['data'])
    except requests.RequestException as e:
        st.error(f"Error fetching Treasury data: {e}")
        return pd.DataFrame()
    except KeyError:
        st.error("Unexpected data format received from Treasury API.")
        return pd.DataFrame()

# Function to extract and clean data
def extract_and_clean_data(dataframe):
    """Extract 'value' column from the dataframe and convert its values to numeric."""
    try:
        dataframe['value'] = pd.to_numeric(dataframe['value'], errors='coerce')
        return dataframe
    except KeyError as e:
        st.error(f"Key error: {e}")
        return pd.DataFrame()

# Function to calculate average yield
def calculate_yield(dataframe, column_name):
    """Calculate the average yield from the specified column in the dataframe."""
    try:
        annual_yield = dataframe[column_name].mean()
        return annual_yield
    except KeyError:
        st.error(f"Column {column_name} not found in dataframe")
        return None

# Function to adjust for inflation
def adjust_for_inflation(nominal_rate, inflation_rate):
    """Adjust the nominal rate for inflation to get the real rate."""
    return nominal_rate - inflation_rate

# Function to fetch CPI data from BLS API
def fetch_cpi_data(api_key, start_year, end_year):
    """Fetch CPI data from the BLS API for the given period."""
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {'Content-type': 'application/json'}
    data = {
        "seriesid": ["CUUR0000SA0"],  # Consumer Price Index for All Urban Consumers: All Items
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": api_key
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data['Results']['series'][0]['data'])
    except requests.RequestException as e:
        st.error(f"Error fetching CPI data: {e}")
        return pd.DataFrame()
    except KeyError:
        st.error("Unexpected data format received from BLS API.")
        return pd.DataFrame()

# Function to calculate inflation rate
def calculate_inflation_rate(cpi_data):
    """Calculate the annual inflation rate based on the percentage change in CPI year-over-year."""
    try:
        cpi_data['value'] = pd.to_numeric(cpi_data['value'], errors='coerce')
        cpi_data['year'] = pd.to_numeric(cpi_data['year'], errors='coerce')
        
        yearly_cpi = cpi_data.groupby('year')['value'].mean()
        inflation_rate = ((yearly_cpi.iloc[-1] - yearly_cpi.iloc[0]) / yearly_cpi.iloc[0]) * 100
        return inflation_rate
    except (KeyError, IndexError) as e:
        st.error(f"Error calculating inflation rate: {e}")
        return None

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
        st.error("Mean-Variance Optimization failed.")
        return None

    return dict(zip(returns.columns, optimized.x))

# Function for Black-Litterman Allocation
def black_litterman_allocation(market_prices, mcaps, cov_matrix, viewdict, tau=0.05):
    """Optimize portfolio using Black-Litterman model."""
    try:
        delta = black_litterman.market_implied_risk_aversion(market_prices.mean())
        prior = black_litterman.market_implied_prior_returns(mcaps, delta, cov_matrix)
        bl = BlackLittermanModel(cov_matrix, pi=prior, absolute_views=viewdict, tau=tau)
        posterior_rets = bl.bl_returns()
        posterior_cov = bl.bl_cov()
        ef = EfficientFrontier(posterior_rets, posterior_cov)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        
        return cleaned_weights, performance
    except Exception as e:
        st.error(f"Black-Litterman Optimization failed: {e}")
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
        st.error("Risk Parity Optimization failed.")
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
        st.error(f"Mean-CVaR Optimization failed: {e}")
        return None

# Function to perform Hierarchical Risk Parity Optimization
def hierarchical_risk_parity_optimization(returns, linkage_method='single'):
    """Optimize portfolio using hierarchical risk parity (HRP) method."""
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()

    # Ensure covariance matrix does not contain NaN values
    if cov_matrix.isnull().values.any():
        st.error("Covariance matrix contains NaN values. HRP optimization cannot proceed.")
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
        st.error("HRP optimization resulted in NaN values for weights.")
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
                left_cov = cov_matrix.iloc[left_cluster, left_cluster].mean().mean()
                right_cov = cov_matrix.iloc[right_cluster, right_cluster].mean().mean()

                # Allocate weights based on inverse variance
                alloc_factor = 1 - left_cov / (left_cov + right_cov)
                weights[left_cluster] *= alloc_factor
                weights[right_cluster] *= (1 - alloc_factor)

                # Add new clusters to the list
                new_clusters.append(left_cluster)
                new_clusters.append(right_cluster)

        cluster_items = new_clusters

    # Normalize the weights to ensure they sum to 1
    return weights / weights.sum()

# Function to calculate portfolio performance
def calculate_portfolio_performance(weights, returns, risk_free_rate=0.0):
    """
    Calculate the expected annual return, annual volatility, and Sharpe ratio for a portfolio.
    """
    portfolio_return = np.dot(weights, returns.mean()) * 252  # Assuming 252 trading days in a year
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualize volatility
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

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
            st.error(f"Error with parameters {params}: {e}")
            continue

    return best_params

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
        title='Cumulative Returns',
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
    labels = weights.keys()
    sizes = weights.values()
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
        st.error("Total weight cannot be zero. Please assign positive weights.")
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
        "Enter the total USD amount to allocate:",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )

    usd_allocation = {stock: allocation * total_usd for stock, allocation in avg_allocation.items()}

    # Display average allocation
    st.markdown("### Average Allocation for Each Stock:")
    allocation_df = pd.DataFrame({
        'Stock': avg_allocation.keys(),
        'Allocation (%)': avg_allocation.values(),
        'Allocation (USD)': usd_allocation.values()
    }).round(2)
    st.table(allocation_df)

    # Plot average allocation
    plot_asset_allocation(avg_allocation, "Average Asset Allocation")

# Function to plot autocorrelation
def plot_autocorrelation(data, title, xlabel, ylabel, save_path=None):
    """Plot autocorrelation using Plotly."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_acf(data, ax=ax, lags=20)
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        plt.tight_layout()

        st.pyplot(fig)

        if save_path:
            plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_')}.png"))
        plt.close(fig)
    except Exception as e:
        st.error(f"Error plotting autocorrelation: {e}")

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
        "Enter Ticker Symbols (comma-separated)",
        value="AAPL, MSFT"
    )
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.date.today() - datetime.timedelta(days=365),
        min_value=datetime.date(1900, 1, 1),
        max_value=datetime.date.today()
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.date.today(),
        min_value=start_date,
        max_value=datetime.date.today()
    )
    
    alpha_vantage_api_key = st.sidebar.text_input(
        "Enter your Alpha Vantage API Key",
        type="password",
        help="Required for fetching Treasury data."
    )
    
    adjust_inflation = st.sidebar.checkbox(
        "Adjust for Inflation",
        value=False,
        help="Fetch CPI data to adjust nominal rates for inflation."
    )
    
    if adjust_inflation:
        bls_api_key = st.sidebar.text_input(
            "Enter your BLS API Key",
            type="password",
            help="Required for fetching CPI data."
        )
    
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
        "Select Optimization Techniques:",
        options=list(techniques_options.values()),
        default=list(techniques_options.values())  # Select all by default
    )
    
    # Run Analysis Button
    run_analysis = st.sidebar.button("🚀 Run Analysis")

    if run_analysis:
        if not tickers:
            st.error("❌ Please enter at least one ticker symbol.")
            return
        if not alpha_vantage_api_key:
            st.error("❌ Please provide a valid Alpha Vantage API Key.")
            return
        if adjust_inflation and not bls_api_key:
            st.error("❌ Please provide a valid BLS API Key for inflation adjustment.")
            return
        
        headers = {'User-Agent': 'PortfolioOptimizationApp/1.0'}
        console_output = []
        
        # Validate Tickers
        with st.spinner("Validating tickers..."):
            valid_tickers = []
            for ticker in tickers:
                try:
                    data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
                    if not data.empty:
                        valid_tickers.append(ticker)
                    else:
                        console_output.append(f"Invalid or unavailable ticker: {ticker}")
                except Exception as e:
                    console_output.append(f"Failed to download {ticker}: {str(e)}")
            if not valid_tickers:
                st.error("❌ No valid tickers found. Please check your inputs.")
                return
            else:
                if len(valid_tickers) < len(tickers):
                    st.warning(f"⚠️ Some tickers were invalid and have been removed: {set(tickers) - set(valid_tickers)}")

        # Fetch Market Data
        with st.spinner("Fetching market data..."):
            market_prices, mcaps = fetch_data(valid_tickers, headers)
            if market_prices.empty:
                st.error("❌ Failed to fetch market data.")
                return
            else:
                st.success("✅ Market data fetched successfully.")
        
        # Calculate Returns
        returns = calculate_daily_returns(market_prices)

        # Fetch Treasury Data
        with st.spinner("Fetching Treasury data..."):
            treasury_api_url = "https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year"
            treasury_data = fetch_treasury_data(treasury_api_url, alpha_vantage_api_key)
            if treasury_data.empty:
                st.error("❌ Failed to fetch Treasury data.")
                return
            else:
                st.success("✅ Treasury data fetched successfully.")
        
        # Extract and clean Treasury data
        treasury_clean = extract_and_clean_data(treasury_data)
        if treasury_clean.empty:
            st.error("❌ Failed to clean Treasury data.")
            return
        nominal_rate = calculate_yield(treasury_clean, 'value')
        if nominal_rate is None:
            st.error("❌ Failed to calculate nominal rate.")
            return
        else:
            st.write(f"**Nominal Treasury Yield (10-year):** {nominal_rate:.2f}%")
        
        # Adjust for Inflation if selected
        if adjust_inflation:
            with st.spinner("Fetching CPI data..."):
                start_year = start_date.year
                end_year = end_date.year
                cpi_data = fetch_cpi_data(bls_api_key, start_year, end_year)
                if cpi_data.empty:
                    st.error("❌ Failed to fetch CPI data.")
                    return
                else:
                    st.success("✅ CPI data fetched successfully.")
            
            # Extract and clean CPI data
            cpi_clean = extract_and_clean_data(cpi_data)
            if cpi_clean.empty:
                st.error("❌ Failed to clean CPI data.")
                return
            inflation_rate = calculate_inflation_rate(cpi_clean)
            if inflation_rate is None:
                st.error("❌ Failed to calculate inflation rate.")
                return
            else:
                st.write(f"**Inflation Rate (Year-over-Year):** {inflation_rate:.2f}%")
            
            # Adjust nominal rate for inflation
            real_risk_free_rate = adjust_for_inflation(nominal_rate, inflation_rate)
            st.write(f"**Real Risk-Free Rate:** {real_risk_free_rate:.2f}%")
            risk_free_rate = real_risk_free_rate / 100  # Convert to decimal
        else:
            risk_free_rate = nominal_rate / 100  # Convert to decimal
        
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
        
        # Initialize dictionary to store optimized portfolios
        optimized_portfolios = {}
        performance_metrics = {}
        
        # Perform selected optimization techniques
        for technique in selected_techniques:
            st.markdown(f"### 🔍 {technique}")
            with st.spinner(f"Performing {technique}..."):
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
                            'Expected Return (%)': portfolio_return,
                            'Volatility (%)': portfolio_volatility,
                            'Sharpe Ratio': sharpe_ratio
                        }
                        st.success("✅ Mean Variance Optimization completed successfully.")
                        plot_asset_allocation(mv_weights, "Mean Variance Asset Allocation")
                elif technique == "Black-Litterman Model":
                    if not viewdict:
                        st.warning("⚠️ No views provided. Skipping Black-Litterman Optimization.")
                        continue
                    bl_weights, bl_performance = black_litterman_allocation(market_prices, mcaps, returns.cov(), viewdict, tau=0.05)
                    if bl_weights:
                        optimized_portfolios['Black-Litterman'] = bl_weights
                        performance_metrics['Black-Litterman'] = {
                            'Expected Return (%)': bl_performance[0],
                            'Volatility (%)': bl_performance[1],
                            'Sharpe Ratio': bl_performance[2]
                        }
                        st.success("✅ Black-Litterman Optimization completed successfully.")
                        plot_asset_allocation(bl_weights, "Black-Litterman Asset Allocation")
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
                            'Expected Return (%)': portfolio_return,
                            'Volatility (%)': portfolio_volatility,
                            'Sharpe Ratio': sharpe_ratio
                        }
                        st.success("✅ Risk Parity Optimization completed successfully.")
                        plot_asset_allocation(rp_weights, "Risk Parity Asset Allocation")
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
                            'Expected Return (%)': portfolio_return,
                            'Volatility (%)': portfolio_volatility,
                            'Sharpe Ratio': sharpe_ratio
                        }
                        st.success("✅ Mean-CVaR Optimization completed successfully.")
                        plot_asset_allocation(cvar_weights, "Mean-CVaR Asset Allocation")
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
                            'Expected Return (%)': portfolio_return,
                            'Volatility (%)': portfolio_volatility,
                            'Sharpe Ratio': sharpe_ratio
                        }
                        st.success("✅ Hierarchical Risk Parity Optimization completed successfully.")
                        plot_asset_allocation(hrp_weights_dict, "Hierarchical Risk Parity Asset Allocation")
        
        if not optimized_portfolios:
            st.error("❌ No portfolios were successfully optimized.")
            return

        # Plot optimized portfolios
        st.subheader("📈 Optimized Portfolios Cumulative Returns")
        plot_optimized_portfolios(optimized_portfolios, market_prices, start_date, end_date)

        # Display performance metrics
        st.subheader("📊 Portfolio Performance Metrics")
        for method, metrics in performance_metrics.items():
            st.markdown(f"**{method}:**")
            metrics_df = pd.DataFrame(metrics, index=[0])
            st.table(metrics_df)

        # Calculate Weighted Average Allocation
        st.subheader("📊 Weighted Average Allocation")
        calculate_weighted_average_allocation(optimized_portfolios)

        # Grid Search Optimization (Optional)
        st.subheader("🔍 Grid Search Optimization")
        best_params = grid_search_optimization(returns, risk_free_rate, market_prices, mcaps, returns.cov(), viewdict)
        if best_params:
            st.write(f"**Best Parameters:** {best_params}")
        else:
            st.warning("⚠️ Grid search did not find better parameters.")

        # Display Console Output
        if console_output:
            st.subheader("📝 Console Output")
            console_text = '\n'.join(console_output)
            st.text_area("Console Output", value=console_text, height=300)

if __name__ == "__main__":
    main()
