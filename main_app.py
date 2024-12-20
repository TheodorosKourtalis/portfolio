#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:19:58 2024

@author: thodoreskourtales
"""

# 1_Fetch_Raw_Data.py

import streamlit as st

def main():
    st.title("📈 Stock Forecasting ")
    st.markdown("""
        Welcome to the Stock Forecasting App! Use the sidebar to navigate through the steps:
        1. **Fetch Raw Data**
        2. **Clean Data**
        3. **Train Prophet Model**
        4. **Forecast & Anomaly Detection**

        Each step builds upon the previous one. Ensure you complete them in order for optimal results.
    """)

if __name__ == "__main__":
    main()
