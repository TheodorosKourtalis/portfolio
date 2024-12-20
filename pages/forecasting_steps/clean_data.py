#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:20:58 2024

Author: Theodoros Kourtales
"""

# pages/2_Clean_Data.py

import streamlit as st
import pandas as pd
import importlib  # For dynamic page loading

def load_page(page_name):
    """
    Dynamically load the corresponding page module based on the page_name.
    """
    try:
        module = importlib.import_module(f"pages.forecasting_steps.{page_name}")
        # Check if the module has a main() function
        if hasattr(module, "main"):
            module.main()
        else:
            st.error(f"The module {page_name} does not have a `main` function.")
    except ModuleNotFoundError:
        st.error(f"Module not found: {page_name}. Ensure the file exists in the forecasting_steps directory.")
    except Exception as e:
        st.error(f"An error occurred while loading the page: {e}")

def clean_data(data):
    try:
        # Convert 'ds' to datetime and 'y' to numeric
        data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
        data['y'] = pd.to_numeric(data['y'], errors='coerce')
        
        # Drop rows with NaN in 'ds' or 'y'
        data = data.dropna(subset=['ds', 'y'])
        
        # Final validation
        st.write("**Cleaned Data Types:**")
        st.write(data.dtypes)
        
        return data
    except Exception as e:
        st.error(f"Error during data cleaning: {e}")
        return None

def main():
    # Set the current page in session_state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "clean_data"

    st.header("🧹 Step 2: Clean Data")
    
    if 'raw_data' not in st.session_state or 'symbol' not in st.session_state:
        st.warning("No raw data or symbol found. Please complete Step 1: Fetch Raw Data.")
        st.markdown("### Return to the Previous Step:")
        if st.button("Go to Step 1: Fetch Raw Data"):
            st.session_state["current_page"] = "fetch_raw_data"
            st.experimental_rerun()  # Rerun to load the new page
        return
    
    data = st.session_state['raw_data']
    symbol = st.session_state['symbol']
    
    st.write("**Data Before Cleaning:**")
    st.dataframe(data.head())
    
    clean_button = st.button("Clean Data")
    
    if clean_button:
        with st.spinner("Cleaning data..."):
            cleaned_data = clean_data(data)
            if cleaned_data is not None and not cleaned_data.empty:
                st.success("Data cleaned successfully!")
                st.write("**Data After Cleaning:**")
                st.dataframe(cleaned_data.head())
                
                # Save cleaned data to session_state
                st.session_state['cleaned_data'] = cleaned_data
                
                # Download Cleaned Data
                st.download_button(
                    label="Download Cleaned Data",
                    data=cleaned_data.to_csv(index=False).encode('utf-8'),
                    file_name=f"{symbol}_cleaned_data.csv",
                    mime="text/csv"
                )
            else:
                st.error("Data cleaning failed. Please check the raw data.")
    
    # Navigation buttons
    st.markdown("---")
    
    # Show "Return to Previous Step" button
    st.markdown("### Navigate Between Steps:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous Step: Fetch Raw Data"):
            st.session_state["current_page"] = "fetch_raw_data"
            st.experimental_rerun()  # Rerun to load the new page
    
    with col2:
        if "cleaned_data" in st.session_state and st.button("Next Step: Train Prophet Model"):
            st.session_state["current_page"] = "train_prophet"
            st.experimental_rerun()  # Rerun to load the new page

if __name__ == "__main__":
    main()
