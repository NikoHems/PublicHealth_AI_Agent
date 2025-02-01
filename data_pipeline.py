import requests
from datetime import datetime
import math
import statistics

# Base URL for the disease.sh API
BASE_URL = "https://disease.sh/v3/covid-19"

def get_global_data():
    """
    Retrieves global COVID-19 data.
    
    Returns:
        dict: JSON object containing global statistics.
    """
    url = f"{BASE_URL}/all"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error retrieving global data: {response.status_code}")

def get_country_data(country):
    """
    Retrieves COVID-19 data for a specific country.
    
    Parameters:
        country (str): Name of the country, e.g., "Germany" or "USA".
        
    Returns:
        dict: JSON object containing COVID-19 statistics for the specified country.
    """
    url = f"{BASE_URL}/countries/{country}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error retrieving data for {country}: {response.status_code}")

def get_historical_data(country="all", lastdays=30):
    """
    Retrieves historical COVID-19 data.
    
    Parameters:
        country (str): Country name ("all" for global data, or a specific country like "Germany").
        lastdays (int): Number of past days for which data should be retrieved.
        
    Returns:
        dict: JSON object containing the historical data.
    """
    url = f"{BASE_URL}/historical/{country}?lastdays={lastdays}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error retrieving historical data: {response.status_code}")

def compute_daily_new_cases(cases_timeline):
    """
    Computes daily new cases from cumulative cases data.
    
    Parameters:
        cases_timeline (dict): Dictionary with dates as keys and cumulative case counts as values.
                               Expected date format: "MM/DD/YY".
                               
    Returns:
        dict: Dictionary with dates (starting from the second entry) and the computed daily new cases.
    """
    # Sort the dates based on the datetime
    sorted_dates = sorted(cases_timeline.keys(), key=lambda d: datetime.strptime(d, "%m/%d/%y"))
    daily_new = {}
    previous_count = None
    for date in sorted_dates:
        current_count = cases_timeline[date]
        if previous_count is None:
            # Cannot compute daily new cases for the first date; set as None or 0
            daily_new[date] = 0
        else:
            daily_new[date] = current_count - previous_count
        previous_count = current_count
    return daily_new

def compute_moving_average(data_list, window=7):
    """
    Computes the moving average for a list of numerical values.
    
    Parameters:
        data_list (list): List of numerical values.
        window (int): Window size for the moving average.
        
    Returns:
        list: List of moving average values. The first (window-1) entries will be None to indicate insufficient data.
    """
    moving_avg = []
    for i in range(len(data_list)):
        if i < window - 1:
            moving_avg.append(None)
        else:
            window_values = data_list[i-window+1:i+1]
            avg = sum(window_values) / window
            moving_avg.append(avg)
    return moving_avg

def compute_average_growth_rate(daily_cases):
    """
    Computes the average daily growth rate based on daily new cases.
    
    Parameters:
        daily_cases (list): List of daily new cases.
        
    Returns:
        float: The average daily growth rate (as a fraction), calculated over days where previous day's cases are > 0.
    """
    growth_rates = []
    for i in range(1, len(daily_cases)):
        prev = daily_cases[i-1]
        curr = daily_cases[i]
        # Avoid division by zero and ignore days with no cases as a base
        if prev > 0:
            growth_rate = (curr - prev) / prev
            growth_rates.append(growth_rate)
    if growth_rates:
        return statistics.mean(growth_rates)
    else:
        return 0.0

def compute_doubling_time(average_growth_rate):
    """
    Computes the doubling time based on the average daily growth rate.
    
    Parameters:
        average_growth_rate (float): Average daily growth rate (as a fraction).
        
    Returns:
        float: Doubling time in days. Returns math.inf if the growth rate is zero or negative.
    """
    if average_growth_rate > 0:
        return math.log(2) / average_growth_rate
    else:
        return math.inf

if __name__ == "__main__":
    # Retrieve historical data for Germany for the last 30 days
    try:
        historical_data = get_historical_data("Germany", lastdays=30)
        # The historical data for a country is typically under the "timeline" key
        timeline = historical_data.get("timeline", {})
        cases_timeline = timeline.get("cases", {})
        if not cases_timeline:
            raise Exception("No cases timeline data found.")
        
        # Compute daily new cases from the cumulative data
        daily_new_cases = compute_daily_new_cases(cases_timeline)
        print("Daily New Cases:")
        for date, new_cases in daily_new_cases.items():
            print(f"{date}: {new_cases}")
        
        # Convert daily new cases to a list (in sorted order) for further computations
        sorted_dates = sorted(daily_new_cases.keys(), key=lambda d: datetime.strptime(d, "%m/%d/%y"))
        daily_cases_list = [daily_new_cases[date] for date in sorted_dates]
        
        # Compute a 7-day moving average of daily new cases
        moving_avg = compute_moving_average(daily_cases_list, window=7)
        print("\n7-Day Moving Average of Daily New Cases:")
        for date, avg in zip(sorted_dates, moving_avg):
            print(f"{date}: {avg}")
        
        # Compute average growth rate over the period
        avg_growth_rate = compute_average_growth_rate(daily_cases_list)
        print(f"\nAverage Daily Growth Rate: {avg_growth_rate:.4f}")
        
        # Compute doubling time based on the average growth rate
        doubling_time = compute_doubling_time(avg_growth_rate)
        if doubling_time == math.inf:
            print("Doubling Time: Infinity (no growth)")
        else:
            print(f"Doubling Time: {doubling_time:.2f} days")
        
    except Exception as e:
        print(e)
