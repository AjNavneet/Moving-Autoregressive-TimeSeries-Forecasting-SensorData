# Importing required libraries
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

# Create a class for the AutoRegressive (AR) model
def AR_model(df):
    # Create AR model
    order = 1  # Order of the AR model
    model = AutoReg(df['IOT_Sensor_Reading'], lags=order)
    
    # Fit the AR model
    model_fit = model.fit()
    
    # Get the fitted values
    fitted_values = model_fit.fittedvalues
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(df['IOT_Sensor_Reading'][:-1], fitted_values))
    
    # Redirect the standard output to a log file
    with open('../output/' + 'log.txt', 'a') as f:
        sys.stdout = f
        print("***************************AR1 Model*******************************")
        print(f"RMSE of AR1 model: {rmse}")
        print(model_fit.summary())
    sys.stdout = sys.__stdout__  # Restore the standard output
    
    # Plot the original data and the fitted values
    plt.plot(df['IOT_Sensor_Reading'], label='Original Data')
    plt.plot(fitted_values, label='Fitted Values')
    plt.legend()
    plt.title('First-Order AR Model')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('../output/' + 'AR_Model.png')  # Save the plot
    plt.show()
    
    # Return model summary
    return model_fit.summary()
