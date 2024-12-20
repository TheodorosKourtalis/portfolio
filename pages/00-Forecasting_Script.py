import streamlit as st

def main():
    st.title("📈 Forecasting Workflow")
    st.markdown("""
    Welcome to the **Forecasting Workflow**! Follow the steps below to generate stock price forecasts:
    
    ### Steps:
    1. **Fetch Raw Data:** Retrieve stock data from Yahoo Finance.
    2. **Clean Data:** Preprocess the data to prepare it for modeling.
    3. **Train Prophet Model:** Train a forecasting model using Facebook Prophet.
    4. **Forecast:** Generate future forecasts and visualize them interactively.
    """)

    st.markdown("---")
    st.markdown("### Select a Step to Proceed:")

    # Custom Navigation
    if st.button("Step 1: Fetch Raw Data"):
        st.session_state["current_page"] = "fetch_raw_data"

    if st.button("Step 2: Clean Data"):
        st.session_state["current_page"] = "clean_data"

    if st.button("Step 3: Train Prophet Model"):
        st.session_state["current_page"] = "train_prophet"

    if st.button("Step 4: Forecast"):
        st.session_state["current_page"] = "forecast"

    # Load the corresponding page
    if "current_page" in st.session_state:
        if st.session_state["current_page"] == "fetch_raw_data":
            import pages.forecasting_steps.01_Fetch_Raw_Data
        elif st.session_state["current_page"] == "clean_data":
            import pages.forecasting_steps.02_Clean_Data
        elif st.session_state["current_page"] == "train_prophet":
            import pages.forecasting_steps.03_Train_Prophet
        elif st.session_state["current_page"] == "forecast":
            import pages.forecasting_steps.04_Forecast

if __name__ == "__main__":
    main()
