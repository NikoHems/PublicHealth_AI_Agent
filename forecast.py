import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# Import functions from the data pipeline module
from data_pipeline import get_historical_data, compute_daily_new_cases

def forecast_arima(daily_cases, forecast_period=7, order=(1,1,1)):
    """
    Fits an ARIMA model to the daily new cases data and forecasts future values.
    
    Parameters:
        daily_cases (list): List of daily new cases (numerical values).
        forecast_period (int): Number of days to forecast.
        order (tuple): The (p, d, q) order of the ARIMA model.
        
    Returns:
        forecast (pd.Series): Forecasted daily new cases for the specified period.
    """
    # Convert the list of daily cases into a pandas Series
    series = pd.Series(daily_cases)
    
    # Fit the ARIMA model with the specified order
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    
    # Forecast the next forecast_period days
    forecast_result = model_fit.forecast(steps=forecast_period)
    
    return forecast_result, model_fit

if __name__ == "__main__":
    # Retrieve historical data for Germany for the last 30 days
    try:
        historical_data = get_historical_data("Germany", lastdays=30)
        timeline = historical_data.get("timeline", {})
        cases_timeline = timeline.get("cases", {})

        if not cases_timeline:
            raise Exception("No cases timeline data found.")

        # Compute daily new cases from the cumulative cases data
        daily_new_cases_dict = compute_daily_new_cases(cases_timeline)

        # Sort dates to create a time-ordered list of daily new cases
        sorted_dates = sorted(daily_new_cases_dict.keys(), key=lambda d: datetime.strptime(d, "%m/%d/%y"))
        daily_cases_list = [daily_new_cases_dict[date] for date in sorted_dates]

        print("Historical Daily New Cases:")
        print(daily_cases_list)

        # Forecast the next 7 days using the ARIMA model
        forecast_period = 7
        forecast_values = forecast_arima(daily_cases_list, forecast_period=forecast_period, order=(1,1,1))

        print(f"\nForecast for the next {forecast_period} days:")
        print(forecast_values)

        # Optional: Plot the historical data and forecast
        dates = [datetime.strptime(d, "%m/%d/%y") for d in sorted_dates]
        forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=forecast_period)

        plt.figure(figsize=(10, 5))
        plt.plot(dates, daily_cases_list, label='Historical Daily New Cases')
        plt.plot(forecast_dates, forecast_values, label='Forecast', linestyle='--', marker='o')
        plt.xlabel("Date")
        plt.ylabel("Daily New Cases")
        plt.title("ARIMA Forecast of Daily New Cases")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
