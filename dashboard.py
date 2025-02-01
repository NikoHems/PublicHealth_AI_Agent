import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import base64

# Import functions from the data pipeline, forecast, XAI, and NLP modules
from data_pipeline import get_global_data, get_country_data, get_historical_data, compute_daily_new_cases
from forecast import forecast_arima
from xai_module import get_arima_explanation
from nlp_module import generate_narrative_report

st.title("COVID-19 Data Dashboard with Forecasting, XAI, and Narrative Report")

# --- Global Data Section ---
st.header("Global COVID-19 Data")
try:
    global_data = get_global_data()
    st.write(global_data)
except Exception as e:
    st.error(f"Error retrieving global data: {e}")

# --- Country-Specific Data Section ---
st.header("Country-Specific COVID-19 Data")
country = st.selectbox("Select a country", ["Germany", "USA", "France"])
try:
    country_data = get_country_data(country)
    st.write(country_data)
except Exception as e:
    st.error(f"Error retrieving country data: {e}")

# --- Historical Data, Forecast, XAI and Narrative Report Section ---
st.header("Historical Data, Forecast, Model Explanation and Narrative Report")
historical_days = st.slider("Select number of historical days", min_value=15, max_value=60, value=30, step=1)

try:
    historical_data = get_historical_data(country, lastdays=historical_days)
    timeline = historical_data.get("timeline", {})
    cases_timeline = timeline.get("cases", {})
    
    if cases_timeline:
        # Compute daily new cases from cumulative data
        daily_new_cases_dict = compute_daily_new_cases(cases_timeline)
        sorted_dates = sorted(daily_new_cases_dict.keys(), key=lambda d: datetime.strptime(d, "%m/%d/%y"))
        daily_cases_list = [daily_new_cases_dict[date] for date in sorted_dates]
        historical_dates = [datetime.strptime(date, "%m/%d/%y") for date in sorted_dates]
        
        # Create DataFrame for historical data and display chart
        hist_df = pd.DataFrame({
            "Date": historical_dates,
            "Daily New Cases": daily_cases_list
        }).set_index("Date")
        st.subheader("Historical Daily New Cases")
        st.line_chart(hist_df)
        
        # --- Forecasting Section ---
        st.subheader("Forecast")
        forecast_period = st.slider("Select forecast period (days)", min_value=3, max_value=14, value=7, step=1)
        # Generate forecast and get fitted model from forecast.py
        forecast_values, model_fit = forecast_arima(daily_cases_list, forecast_period=forecast_period, order=(1,1,1))
        forecast_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=forecast_period)
        
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecast": forecast_values
        }).set_index("Date")
        
        # Combine historical and forecast data for visualization
        combined_df = pd.concat([hist_df, forecast_df], axis=1)
        st.line_chart(combined_df)
        st.write("Forecasted Values", forecast_df)
        
        # --- XAI Section ---
        st.subheader("ARIMA Model Explanation")
        summary_text, residual_plot = get_arima_explanation(model_fit)
        st.text_area("ARIMA Model Summary", summary_text, height=300)
        st.image(f"data:image/png;base64,{residual_plot}", caption="Residual Plot")
        
        # --- NLP: Narrative Report Section ---
        st.subheader("Narrative Report")
        report = generate_narrative_report(daily_cases_list, forecast_values)
        st.write(report)
        
    else:
        st.warning("No historical cases data available.")
        
except Exception as e:
    st.error(f"Error retrieving historical data: {e}")
