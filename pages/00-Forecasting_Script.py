import streamlit as st

def load_page(page_name):
    """
    Dynamically load the corresponding page module based on the page_name.
    """
    if page_name == "fetch_raw_data":
        import pages.forecasting_steps.fetch_raw_data
    elif page_name == "clean_data":
        import pages.forecasting_steps.clean_data
    elif page_name == "train_prophet":
        import pages.forecasting_steps.train_prophet
    elif page_name == "forecast":
        import pages.forecasting_steps.forecast
    else:
        st.error("Page not found! Please select a valid step.")

def main():
    # Set the title and introduction
    st.title("📈 Forecasting Workflow")
    st.markdown("""
    Welcome to the **Forecasting Workflow**! Follow the steps below to generate stock price forecasts:
    
    ### Steps:
    1. **Fetch Raw Data:** Retrieve stock data from Yahoo Finance.
    2. **Clean Data:** Preprocess the data to prepare it for modeling.
    3. **Train Prophet Model:** Train a forecasting model using Facebook Prophet.
    4. **Forecast:** Generate future forecasts and visualize them interactively.
    """)

    # Horizontal separator
    st.markdown("---")
    st.markdown("### Select a Step to Proceed:")

    # Custom navigation buttons to select a step
    if st.button("Step 1: Fetch Raw Data"):
        st.session_state["current_page"] = "fetch_raw_data"

    if st.button("Step 2: Clean Data"):
        st.session_state["current_page"] = "clean_data"

    if st.button("Step 3: Train Prophet Model"):
        st.session_state["current_page"] = "train_prophet"

    if st.button("Step 4: Forecast"):
        st.session_state["current_page"] = "forecast"

    # Dynamically load the selected page
    if "current_page" in st.session_state:
        load_page(st.session_state["current_page"])

if __name__ == "__main__":
    main()
