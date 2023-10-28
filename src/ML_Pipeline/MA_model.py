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

def MA_model(df):
    # Set the index to 'time' column
    df.set_index('time', inplace=True)

    # Create Moving Average (MA) model
    order = (0, 0, 1)  # (p, d, q) order of the model
    model = ARIMA(df['IOT_Sensor_Reading'], order=order)
    
    # Fit the MA model
    model_fit = model.fit()
    
    # Get the fitted values
    fitted_values = model_fit.fittedvalues
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(df['IOT_Sensor_Reading'], fitted_values))
    
    # Redirect the standard output to a log file
    with open('../output/' + 'log.txt', 'a') as f:
        sys.stdout = f
        print("***************************MA Model*******************************")
        print(f"RMSE of MA model: {rmse}")
        print(model_fit.summary())
    sys.stdout = sys.__stdout__  # Restore the standard output
    
    # Plot the original data and the fitted values
    plt.plot(df['IOT_Sensor_Reading'], label='Original Data')
    plt.plot(fitted_values, label='Fitted Values')
    plt.legend()
    plt.title('Moving Average Model')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('../output/' + 'MA_Model.png')  # Save the plot
    plt.show()
    
    # Return model summary
    return model_fit.summary()
