# Importing required libraries
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

def autocovariance(df):
    # Set the index to 'time' column
    df.set_index('time', inplace=True)

    # Print the DataFrame for inspection
    print("###############################")
    print(df)

    # Calculate the autocovariance matrix
    autocov_matrix = np.cov(df, rowvar=False)

    # Print the autocovariance matrix
    print("Autocovariance Matrix:")
    print(autocov_matrix)

    return autocov_matrix
