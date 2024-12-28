#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Analysis and Forecasting Application

Created on Fri Dec 20 22:19:58 2024

@author: thodoreskourtales
"""

import streamlit as st
from streamlit_extras.switch_page_button import switch_page


def main():
    st.title("📊 Stock Analysis and Forecasting Application")
    st.markdown("""
    Welcome to the **Stock Analysis and Forecasting Application**!
    
    Navigate through the app to explore various features:
    - **Main App**: Overview and navigation.
    - **Forecasting Script**: Predict stock prices using advanced forecasting models.
    - **Portfolio Allocation**: Optimize your portfolio with multiple techniques.
    - **Sec Analysis**: Conduct detailed security analysis.
    """)

    st.subheader("📍 Select a Module to Get Started:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Main App"):
            switch_page("Main_App")

    with col2:
        if st.button("Forecasting Script"):
            switch_page("Forecasting_Script")

    with col3:
        if st.button("Portfolio Allocation"):
            switch_page("Portfolio_Allocation")

    with col4:
        if st.button("Sec Analysis"):
            switch_page("Sec_Analysis")


if __name__ == "__main__":
    main()
