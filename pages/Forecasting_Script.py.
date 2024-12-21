import streamlit as st
import importlib

def load_page(page_name):
    """
    Dynamically load the corresponding page module based on the page_name.
    """
    try:
        module = importlib.import_module(f"pages.forecasting_steps.{page_name}")
        module.main()  # Call the main function in the loaded module (if it exists)
    except ModuleNotFoundError as e:
        st.error(f"Module not found: {page_name}. Ensure the file exists in the forecasting_steps directory.")
    except AttributeError:
        st.error(f"The module {page_name} does not have a `main` function.")

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
