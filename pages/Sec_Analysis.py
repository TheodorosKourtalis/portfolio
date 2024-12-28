

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEC Filing Scraper and Analysis - Streamlit Version

Author: Thodoreskourtales
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
import warnings
from io import StringIO
import tempfile

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch data from URL
def fetch_data(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to retrieve data from {url}: {e}")
        return None

# Function to compute CAGR
def calculate_cagr(start_value, end_value, periods):
    return ((end_value / start_value) ** (1 / periods) - 1) * 100

# Function to fetch and process data for a specific ticker
def get_company_data(ticker, headers):
    company_data = fetch_data("https://www.sec.gov/files/company_tickers.json", headers)
    
    if not company_data:
        logging.error("Failed to retrieve company tickers.")
        return None, None

    # Find the CIK for the given ticker
    cik = None
    for key, value in company_data.items():
        if value['ticker'] == ticker:
            cik = value['cik_str']
            break

    if cik is None:
        logging.error(f"{ticker} not found in the SEC company tickers list.")
        return None, None

    # Add leading zeros to CIK
    cik = str(cik).zfill(10)

    # Get company specific filing metadata
    filing_metadata = fetch_data(f'https://data.sec.gov/submissions/CIK{cik}.json', headers)
    if not filing_metadata:
        logging.error("Failed to retrieve filing metadata.")
        return None, None

    # Parse the metadata
    filings = filing_metadata.get('filings', {}).get('recent', {})
    all_forms = pd.DataFrame.from_dict(filings)
    
    # Get company facts data
    company_facts = fetch_data(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json', headers)
    if not company_facts:
        logging.error("Failed to retrieve company facts data.")
        return None, None

    return all_forms, company_facts

# Function to calculate financial ratios
def calculate_ratios(company_facts):
    ratios = {}
    us_gaap = company_facts.get('facts', {}).get('us-gaap', {})

    def get_value(key):
        values = us_gaap.get(key, {}).get('units', {}).get('USD', [])
        return values[-1]['val'] if values else None

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

    market_price_per_share = 100  # Placeholder
    earnings_per_share = net_income / 100 if net_income is not None else None  # Placeholder
    book_value_per_share = shareholder_equity / 100 if shareholder_equity is not None else None  # Placeholder
    annual_dividends_per_share = 2  # Placeholder

    try:
        # Liquidity Ratios
        ratios['Current Ratio'] = current_assets / current_liabilities if current_assets and current_liabilities else None
        ratios['Quick Ratio'] = (current_assets - inventory) / current_liabilities if current_assets and inventory and current_liabilities else None
        ratios['Cash Ratio'] = cash_and_cash_equivalents / current_liabilities if cash_and_cash_equivalents and current_liabilities else None
        ratios['Operating Cash Flow Ratio'] = operating_cash_flow / current_liabilities if operating_cash_flow and current_liabilities else None
        ratios['Working Capital'] = current_assets - current_liabilities if current_assets and current_liabilities else None

        # Profitability Ratios
        ratios['ROA'] = net_income / total_assets if net_income and total_assets else None
        ratios['ROE'] = net_income / shareholder_equity if net_income and shareholder_equity else None
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
        ratios['DSCR'] = ebit / total_debt_service if ebit and total_debt_service else None

        # Efficiency Ratios
        ratios['Asset Turnover Ratio'] = net_sales / total_assets if net_sales and total_assets else None
        ratios['Inventory Turnover Ratio'] = net_sales / inventory if net_sales and inventory else None
        ratios['Receivables Turnover Ratio'] = net_sales / inventory if net_sales and inventory else None

        # Valuation Ratios
        ratios['P/E Ratio'] = market_price_per_share / earnings_per_share if market_price_per_share and earnings_per_share else None
        ratios['P/B Ratio'] = market_price_per_share / book_value_per_share if market_price_per_share and book_value_per_share else None
        ratios['Dividend Yield'] = (annual_dividends_per_share / market_price_per_share) * 100 if market_price_per_share and annual_dividends_per_share else None

    except ZeroDivisionError as e:
        logging.error(f"Error calculating ratios: {e}")
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

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle(f"Financial Ratios for {company_name} by Categories", fontsize=20, weight='bold')

    for ax, (category, data) in zip(axes.flatten(), categories.items()):
        names = [name for name, value in data.items() if value is not None]
        values = [value for value in data.values() if value is not None]

        if values:  # Only plot if there are values
            sns.barplot(x=names, y=values, ax=ax)
            ax.set_title(category, fontsize=16, weight='bold')
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel("Value")

            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_plots and plot_save_path:
        plt.savefig(os.path.join(plot_save_path, f"{company_name}_financial_ratios_by_category.png"))
    st.pyplot(fig)
    plt.close(fig)

    # Display ratios in tables
    for category, data in categories.items():
        df = pd.DataFrame(list(data.items()), columns=['Ratio', 'Value']).dropna()
        if not df.empty:
            st.subheader(category)
            st.table(df)

    # Log ratios
    for ratio, value in ratios.items():
        message = f"{company_name} - {ratio}: {value:.2f}" if value is not None else f"{company_name} - {ratio}: N/A"
        console_output.append(message)

# Function to plot historical data for a given metric
def plot_historical_data(data, metric, company_name, save_plots=False, plot_save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data['end'], data[metric], marker='o', linestyle='-')
    ax.set_title(f"{company_name} - Historical {metric}", fontsize=16, weight='bold')
    ax.set_xlabel("End Date", fontsize=14, weight='bold')
    ax.set_ylabel(metric, fontsize=14, weight='bold')
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_plots and plot_save_path:
        plt.savefig(os.path.join(plot_save_path, f"{company_name}_historical_{metric}.png"))
    st.pyplot(fig)
    plt.close(fig)

# Function to plot timeseries
def plot_timeseries(x, y, title, xlabel, ylabel, legend, marker='o', linestyle='-', color='b', save_path=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(x, y, marker=marker, linestyle=linestyle, color=color, label=legend, linewidth=2, markersize=6)
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel(xlabel, fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}.png"))
    st.pyplot(fig)
    plt.close(fig)

# Function to plot histogram
def plot_histogram(data, bins, title, xlabel, ylabel, color='b', alpha=0.7, save_path=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='k', linewidth=1.2)
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel(xlabel, fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}.png"))
    st.pyplot(fig)
    plt.close(fig)

# Function to plot boxplot
def plot_boxplot(data, title, xlabel, ylabel, save_path=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red'))
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel(xlabel, fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}.png"))
    st.pyplot(fig)
    plt.close(fig)

# Function to plot scatter plot with trend line
def plot_scatter_with_trend(x, y, title, xlabel, ylabel, legend, save_path=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.regplot(x=x, y=y, marker='o', scatter_kws={'s': 100, 'alpha': 0.7}, ax=ax)
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel(xlabel, fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    ax.legend([legend], fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}.png"))
    st.pyplot(fig)
    plt.close(fig)

# Function to plot heatmap
def plot_heatmap(corr_matrix, title, xlabel, ylabel, save_path=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel(xlabel, fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}.png"))
    st.pyplot(fig)
    plt.close(fig)

# Function to handle console output
def print_and_save(output, save_path=None):
    if save_path:
        with open(save_path, 'a') as f:
            f.write(output + '\n')
    return output  # Return the message to display in Streamlit

# Function to display console outputs
def display_console_output(console_output):
    st.text('\n'.join(console_output))

# Function to plot autocorrelation
def plot_autocorrelation(data, title, xlabel, ylabel, save_path=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    pd.plotting.autocorrelation_plot(data, ax=ax)
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel(xlabel, fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}.png"))
    st.pyplot(fig)
    plt.close(fig)

# Main Streamlit App
def main():
    st.title("SEC Filing Scraper and Analysis")
    st.markdown("""
    This application fetches and analyzes SEC filings for specified companies. 
    Enter the ticker symbols of the companies you wish to analyze, provide your 
    User-Agent for the SEC EDGAR API, and explore various financial ratios and visualizations.
    """)

    # Sidebar Inputs
    st.sidebar.header("Input Parameters")
    
    tickers_input = st.sidebar.text_input(
        "Enter Ticker Symbols (comma-separated)",
        value="AAPL, MSFT"
    )
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    
    user_agent = st.sidebar.text_input(
        "Enter Your User-Agent for SEC EDGAR API",
        value="Your Name Contact@domain.com"
    )
    
    save_output = st.sidebar.checkbox(
        "Save Console Output and Plots",
        value=False
    )
    
    output_directory = None
    if save_output:
        output_directory = st.sidebar.text_input(
            "Enter Directory to Save Outputs",
            value=os.path.join(os.getcwd(), "SEC_outputs")
        )
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory, exist_ok=True)
                st.sidebar.success(f"Created directory: {output_directory}")
            except Exception as e:
                st.sidebar.error(f"Failed to create directory: {e}")
                save_output = False  # Disable saving if directory creation fails

    if st.sidebar.button("Run Analysis"):
        if not tickers:
            st.error("Please enter at least one ticker symbol.")
            return
        if not user_agent:
            st.error("Please provide a valid User-Agent string.")
            return
        
        headers = {'User-Agent': user_agent}
        console_output = []

        # Initialize a temporary directory if not saving
        if not save_output:
            temp_dir = tempfile.mkdtemp()
            plot_save_path = temp_dir
        else:
            plot_save_path = output_directory

        for ticker in tickers:
            st.subheader(f"Processing {ticker}")
            all_forms, company_facts = get_company_data(ticker, headers)
            
            if all_forms is None or company_facts is None:
                st.error(f"Failed to retrieve data for {ticker}. Check console logs for details.")
                continue

            ratios = calculate_ratios(company_facts)
            if ratios:
                log_and_plot_ratios(ratios, ticker, save_output, plot_save_path, console_output)
            else:
                st.warning(f"No ratios calculated for {ticker}.")

            # Data extraction and validation
            assets_10Q = pd.DataFrame(company_facts['facts']['us-gaap'].get('Assets', {}).get('units', {}).get('USD', []))
            if assets_10Q.empty:
                st.warning(f"No Assets data found for {ticker}.")
                continue

            assets_10Q = assets_10Q[['end', 'val']]
            assets_10Q['end'] = pd.to_datetime(assets_10Q['end'])
            assets_10Q = assets_10Q.sort_values('end').reset_index(drop=True)

            if assets_10Q['end'].duplicated().any():
                st.warning("Duplicate dates found and will be dropped.")
                assets_10Q = assets_10Q.drop_duplicates(subset='end')
            if assets_10Q['end'].isnull().any():
                st.warning("Missing dates found and will be filled.")
                assets_10Q['end'] = assets_10Q['end'].fillna(method='ffill')

            # Calculate and log statistics
            console_output.append(f"\n{'='*20} Summary Statistics for {ticker} {'='*20}")
            summary_stats = assets_10Q['val'].describe()
            console_output.append(summary_stats.to_string())
            st.write("**Summary Statistics:**")
            st.table(summary_stats.to_frame())

            # Year-over-Year (YoY) Growth Rate
            assets_10Q['YoY_Growth'] = assets_10Q['val'].pct_change(4) * 100
            console_output.append(f"\n{'='*20} Year-over-Year (YoY) Growth Rates for {ticker} {'='*20}")
            yoy_growth = assets_10Q[['end', 'YoY_Growth']].dropna()
            console_output.append(yoy_growth.to_string(index=False))
            st.write("**Year-over-Year (YoY) Growth Rates:**")
            st.table(yoy_growth)

            # Moving Average (4 periods)
            assets_10Q['Rolling_Mean'] = assets_10Q['val'].rolling(window=4).mean()

            # Rolling Standard Deviation (4 periods)
            assets_10Q['Rolling_Std'] = assets_10Q['val'].rolling(window=4).std()

            # Compound Annual Growth Rate (CAGR)
            if not assets_10Q.empty and len(assets_10Q) > 1:
                start_value = assets_10Q['val'].iloc[0]
                end_value = assets_10Q['val'].iloc[-1]
                periods = (assets_10Q['end'].iloc[-1] - assets_10Q['end'].iloc[0]).days / 365.25
                if periods > 0:
                    cagr = calculate_cagr(start_value, end_value, periods)
                    console_output.append(f"\n{'='*20} Compound Annual Growth Rate (CAGR) for {ticker} {'='*20}")
                    console_output.append(f"CAGR: {cagr:.2f}%")
                    st.write(f"**Compound Annual Growth Rate (CAGR): {cagr:.2f}%**")
                else:
                    st.warning("Insufficient data to calculate CAGR.")
            else:
                st.warning("Insufficient data to calculate CAGR.")

            # Quarterly Growth Rate
            assets_10Q['QoQ_Growth'] = assets_10Q['val'].pct_change() * 100
            console_output.append(f"\n{'='*20} Quarterly Growth Rates for {ticker} {'='*20}")
            qoq_growth = assets_10Q[['end', 'QoQ_Growth']].dropna()
            console_output.append(qoq_growth.to_string(index=False))
            st.write("**Quarterly Growth Rates:**")
            st.table(qoq_growth)

            # Monthly Growth Rate
            assets_10Q['Monthly_Growth'] = assets_10Q['val'].pct_change(1) * 100
            console_output.append(f"\n{'='*20} Monthly Growth Rates for {ticker} {'='*20}")
            monthly_growth = assets_10Q[['end', 'Monthly_Growth']].dropna()
            console_output.append(monthly_growth.to_string(index=False))
            st.write("**Monthly Growth Rates:**")
            st.table(monthly_growth)

            # Rolling Variance
            assets_10Q['Rolling_Var'] = assets_10Q['val'].rolling(window=8).var()

            # Rolling Skewness
            assets_10Q['Rolling_Skew'] = assets_10Q['val'].rolling(window=8).skew()

            # Rolling Kurtosis
            assets_10Q['Rolling_Kurt'] = assets_10Q['val'].rolling(window=8).kurt()

            # Plotting
            with st.expander("📈 View Plots"):
                plot_timeseries(
                    assets_10Q['end'], assets_10Q['val'],
                    title=f'{ticker} 10-Q Asset Values Over Time',
                    xlabel='End Date', ylabel='Asset Value (USD)',
                    legend='Asset Value',
                    color='b',
                    save_path=plot_save_path if save_output else None
                )

                plot_timeseries(
                    assets_10Q['end'], assets_10Q['Rolling_Mean'],
                    title=f'{ticker} 10-Q Rolling Mean of Asset Values',
                    xlabel='End Date', ylabel='Rolling Mean (USD)',
                    legend='4-Period Rolling Mean',
                    linestyle='--', color='g',
                    save_path=plot_save_path if save_output else None
                )

                plot_timeseries(
                    assets_10Q['end'], assets_10Q['YoY_Growth'],
                    title=f'{ticker} Year-over-Year Asset Growth Rate',
                    xlabel='End Date', ylabel='YoY Growth Rate (%)',
                    legend='YoY Growth Rate (%)',
                    color='r',
                    save_path=plot_save_path if save_output else None
                )

                plot_timeseries(
                    assets_10Q['end'], assets_10Q['QoQ_Growth'],
                    title=f'{ticker} Quarterly Asset Growth Rate',
                    xlabel='End Date', ylabel='Quarterly Growth Rate (%)',
                    legend='Quarterly Growth Rate (%)',
                    color='g',
                    save_path=plot_save_path if save_output else None
                )

                plot_timeseries(
                    assets_10Q['end'], assets_10Q['Monthly_Growth'],
                    title=f'{ticker} Monthly Asset Growth Rate',
                    xlabel='End Date', ylabel='Monthly Growth Rate (%)',
                    legend='Monthly Growth Rate (%)',
                    color='b',
                    save_path=plot_save_path if save_output else None
                )

                plot_histogram(
                    assets_10Q['val'], bins=20,
                    title=f'Distribution of {ticker} Asset Values',
                    xlabel='Asset Value (USD)', ylabel='Frequency',
                    save_path=plot_save_path if save_output else None
                )

                plot_boxplot(
                    assets_10Q['QoQ_Growth'].dropna(),
                    title=f'Boxplot of {ticker} Quarterly Growth Rates',
                    xlabel='Quarters', ylabel='Quarterly Growth Rate (%)',
                    save_path=plot_save_path if save_output else None
                )

                plot_timeseries(
                    assets_10Q['end'], assets_10Q['Rolling_Var'],
                    title=f'Rolling Variance of {ticker} Asset Values',
                    xlabel='End Date', ylabel='Rolling Variance',
                    legend='Rolling Variance', color='m',
                    save_path=plot_save_path if save_output else None
                )

                plot_timeseries(
                    assets_10Q['end'], assets_10Q['Rolling_Skew'],
                    title=f'Rolling Skewness of {ticker} Asset Values',
                    xlabel='End Date', ylabel='Rolling Skewness',
                    legend='Rolling Skewness', color='c',
                    save_path=plot_save_path if save_output else None
                )

                plot_timeseries(
                    assets_10Q['end'], assets_10Q['Rolling_Kurt'],
                    title=f'Rolling Kurtosis of {ticker} Asset Values',
                    xlabel='End Date', ylabel='Rolling Kurtosis',
                    legend='Rolling Kurtosis', color='y',
                    save_path=plot_save_path if save_output else None
                )

                plot_autocorrelation(
                    assets_10Q['val'],
                    title=f'Autocorrelation of {ticker} Asset Values',
                    xlabel='Lags', ylabel='Autocorrelation',
                    save_path=plot_save_path if save_output else None
                )

            # Correlation Matrix
            correlation = assets_10Q[['val', 'YoY_Growth']].corr()
            console_output.append(f"\n{'='*20} Correlation Matrix for {ticker} {'='*20}")
            console_output.append(correlation.to_string())
            st.write("**Correlation Matrix:**")
            st.table(correlation)

            with st.expander("🔍 View Correlation Heatmap"):
                plot_heatmap(
                    correlation,
                    title=f'Correlation Matrix for {ticker}',
                    xlabel='Metrics', ylabel='Metrics',
                    save_path=plot_save_path if save_output else None
                )

            # Scatter Plot with Trend Line
            st.write("**Scatter Plot: Asset Value vs YoY Growth Rate**")
            plot_scatter_with_trend(
                assets_10Q['val'], assets_10Q['YoY_Growth'],
                title=f'Correlation between {ticker} Asset Value and YoY Growth Rate',
                xlabel='Asset Value (USD)', ylabel='YoY Growth Rate (%)',
                legend='Trend Line',
                save_path=plot_save_path if save_output else None
            )

            # Historical Data Plots
            historical_data = company_facts.get('facts', {}).get('us-gaap', {})
            metrics_to_plot = ['AssetsCurrent', 'LiabilitiesCurrent', 'NetIncomeLoss', 'SalesRevenueNet']
            for metric in metrics_to_plot:
                if metric in historical_data:
                    data_points = historical_data[metric].get('units', {}).get('USD', [])
                    if data_points:
                        try:
                            dates = [datetime.strptime(dp['end'], '%Y-%m-%d') for dp in data_points]
                            values = [dp['val'] for dp in data_points]
                            historical_df = pd.DataFrame({'end': dates, metric: values})
                            st.write(f"**Historical {metric}:**")
                            st.line_chart(historical_df.set_index('end')[metric])

                            plot_historical_data(
                                historical_df, metric, ticker, save_output, plot_save_path
                            )
                        except Exception as e:
                            st.error(f"Error processing historical data for {metric}: {e}")
                            continue

        # Display Console Output
        if console_output:
            st.subheader("Console Output")
            display_console_output(console_output)

        # Optionally, provide download links for saved outputs
        if save_output:
            st.subheader("Download Outputs")
            # Download console_output.txt
            console_file_path = os.path.join(plot_save_path, 'console_output.txt')
            if os.path.exists(console_file_path):
                with open(console_file_path, 'r') as f:
                    console_content = f.read()
                st.download_button(
                    label="Download Console Output",
                    data=console_content,
                    file_name='console_output.txt',
                    mime='text/plain'
                )
            
            # List saved plot files
            plot_files = [f for f in os.listdir(plot_save_path) if f.endswith('.png')]
            if plot_files:
                st.write("**Plots:**")
                for plot_file in plot_files:
                    file_path = os.path.join(plot_save_path, plot_file)
                    with open(file_path, 'rb') as f:
                        btn = st.download_button(
                            label=f"Download {plot_file}",
                            data=f,
                            file_name=plot_file,
                            mime='image/png'
                        )
            else:
                st.write("No plots were saved.")

if __name__ == "__main__":
    main()