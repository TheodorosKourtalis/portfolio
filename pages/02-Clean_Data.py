#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:20:58 2024

@author: thodoreskourtales
"""

# pages/2_Clean_Data.py

import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page  # Import switch_page for navigation

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
    st.header("🧹 Step 2: Clean Data")
    
    if 'raw_data' not in st.session_state or 'symbol' not in st.session_state:
        st.warning("No raw data or symbol found. Please complete Step 1: Fetch Raw Data.")
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
    
    # Add navigation buttons for next and previous steps
    st.markdown("---")
    st.markdown("### Navigate to the Next Step:")
    
    if 'cleaned_data' in st.session_state:
        if st.button("Next Step: Train Prophet Model"):
            switch_page("train prophet")
    else:
        st.warning("Please clean the data before proceeding to the next step.")
    
    st.markdown("---")
    st.markdown("### Return to the Previous Step:")
    if st.button("Previous Step: Fetch Raw Data"):
        switch_page("fetch raw data")

if __name__ == "__main__":
    main()
