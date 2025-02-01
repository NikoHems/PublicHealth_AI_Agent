import streamlit as st
import pandas as pd
from datetime import datetime
from data_pipeline import get_country_data, get_historical_data, compute_daily_new_cases, compute_moving_average, compute_average_growth_rate, compute_doubling_time

st.title("COVID-19 Data Dashboard")

# Country selection
country = st.selectbox("Select a country", ["Germany", "USA", "France"])

# Retrieve and display country data
try:
    country_data = get_country_data(country)
    st.subheader(f"COVID-19 Data for {country}")
    st.write(country_data)
except Exception as e:
    st.error(f"Error: {e}")

# Retrieve historical data
try:
    days = st.slider("Select number of days for historical data", min_value=15, max_value=60, value=30)
    historical_data = get_historical_data(country, lastdays=days)
    timeline = historical_data.get("timeline", {})
    cases_timeline = timeline.get("cases", {})

    if cases_timeline:
        # Compute daily new cases and 7-day moving average
        daily_new_cases = compute_daily_new_cases(cases_timeline)
        sorted_dates = sorted(daily_new_cases.keys(), key=lambda d: datetime.strptime(d, "%m/%d/%y"))
        daily_cases_list = [daily_new_cases[date] for date in sorted_dates]
        moving_avg = compute_moving_average(daily_cases_list, window=7)
        
        # Create a DataFrame for visualization
        data = pd.DataFrame({
            "Date": pd.to_datetime(sorted_dates, format="%m/%d/%y"),
            "Daily New Cases": daily_cases_list,
            "7-Day Moving Average": moving_avg
        })
        data = data.set_index("Date")
        
        st.subheader("Daily New Cases & 7-Day Moving Average")
        st.line_chart(data)
        
        # Compute and display growth metrics
        avg_growth_rate = compute_average_growth_rate(daily_cases_list)
        doubling_time = compute_doubling_time(avg_growth_rate)
        
        st.write(f"**Average Daily Growth Rate:** {avg_growth_rate:.4f}")
        if doubling_time == float("inf"):
            st.write("**Doubling Time:** Infinity (no growth)")
        else:
            st.write(f"**Doubling Time:** {doubling_time:.2f} days")
    else:
        st.warning("No cases timeline data found.")
        
except Exception as e:
    st.error(f"Error retrieving historical data: {e}")
