def generate_narrative_report(daily_cases, forecast_values):
    """
    Generates a narrative report summarizing historical data and forecast.
    
    Parameters:
        daily_cases (list): List of historical daily new cases.
        forecast_values (pd.Series): Forecasted daily new cases.
        
    Returns:
        str: A narrative report in natural language.
    """
    # Calculate average historical daily new cases
    avg_daily_cases = sum(daily_cases) / len(daily_cases)
    # Calculate the average forecasted cases
    forecast_avg = forecast_values.mean()
    
    # Determine the trend by comparing forecasted average to historical average
    if forecast_avg < avg_daily_cases:
        trend = "decreasing"
    elif forecast_avg > avg_daily_cases:
        trend = "increasing"
    else:
        trend = "stable"
    
    report = (
        f"Based on the historical data, the average daily new cases were approximately {avg_daily_cases:.0f}. "
        f"The forecast for the upcoming period suggests an average of {forecast_avg:.0f} daily new cases, "
        f"indicating a {trend} trend. "
    )
    
    # Additional recommendations or warnings can be added here if desired.
    if trend == "increasing":
        report += "This increase may require proactive measures to mitigate further spread."
    elif trend == "decreasing":
        report += "The decreasing trend indicates that current measures might be effective."
    else:
        report += "The situation appears to be stable at the moment."
        
    return report
