# Importing required libraries
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

def EDA(df):
    # Set the index to 'time' column
    df.set_index('time', inplace=True)

    # Decompose the time series into trend, seasonal, cyclical, and irregular components
    decomp = seasonal_decompose(df['IOT_Sensor_Reading'], model='additive', period=365)
    
    # Extract the components
    trend = decomp.trend
    seasonal = decomp.seasonal
    irregular = decomp.resid
    
    # Plot the components
    plt.subplot(411)
    plt.plot(df['IOT_Sensor_Reading'], label='Original')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(irregular, label='Residuals')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    # Save the plot as 'components.png' in the '../output/' directory
    plt.savefig('../output/' + 'components.png')

    # Display the plot
    return plt.show()
