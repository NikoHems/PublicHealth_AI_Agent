import matplotlib.pyplot as plt
import io
import base64

def get_arima_explanation(model_fit):
    """
    Provides an explanation of the ARIMA model by returning the model summary and a residual plot.
    
    Parameters:
        model_fit: The fitted ARIMA model from statsmodels.
    
    Returns:
        tuple: (summary_text, residual_plot)
            - summary_text (str): A text summary of the model's parameters.
            - residual_plot (str): Base64-encoded PNG image of the residuals plot.
    """
    # Get the model summary as text
    summary_text = model_fit.summary().as_text()
    
    # Create a residual plot
    residuals = model_fit.resid
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(residuals, marker='o', linestyle='-', color='tab:blue')
    ax.set_title("ARIMA Model Residuals")
    ax.set_xlabel("Time")
    ax.set_ylabel("Residuals")
    plt.tight_layout()
    
    # Save the plot to a buffer in PNG format
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    # Encode the image to base64 for easy integration in HTML or Streamlit
    residual_plot = base64.b64encode(buf.read()).decode("utf-8")
    
    return summary_text, residual_plot
