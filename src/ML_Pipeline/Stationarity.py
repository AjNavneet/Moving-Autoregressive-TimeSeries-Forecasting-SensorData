# Importing required libraries
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import sys

def Stationary(df):
    with open('../output/' + 'log.txt', 'a') as f:
        sys.stdout = f
        df.set_index('time', inplace=True)
        
        # Stationarity check using ADF
        # Plot the original time series
        plt.plot(df['IOT_Sensor_Reading'])
        plt.title('Original Time Series')
        
        # Save the plot as 'IOT_Sensor_Reading_plot.png' in the '../output/' directory
        plt.savefig('../output/' + 'IOT_Sensor_Reading_plot.png')
        plt.show()
        
        print("****************Stationary check using ADF method:***********************")
        
        # Test for stationarity using ADF
        result = adfuller(df['IOT_Sensor_Reading'])
        print('ADF Statistic: ', result[0])
        print('p-value: ', result[1])
        print('Critical Values: ')
        for key, value in result[4].items():
            print(key, ": ", value)
        
        # Difference the time series and plot
        diff_data = df.diff().dropna()
        plt.plot(diff_data)
        plt.title('Differenced Time Series')
        
        # Save the differenced plot as 'Diff_IOT_Sensor_Reading_plot.png' in the '../output/' directory
        plt.savefig('../output/' + 'Diff_IOT_Sensor_Reading_plot.png')
        plt.show()
    
        # Test for stationarity of the differenced time series using ADF
        result = adfuller(diff_data['IOT_Sensor_Reading'])
        print('ADF Statistic: ', result[0])
        print('p-value: ', result[1])
        print('Critical Values: ')
        for key, value in result[4].items():
            print(key, ": ", value)
            
        # Stationarity check using KPSS
        print("****************Stationary check using KPSS method:***********************")
    
        # Plot the original time series
        plt.plot(df['IOT_Sensor_Reading'])
        plt.title('Original Time Series')
        plt.show()
        
        # Perform KPSS test
        result = kpss(df['IOT_Sensor_Reading'])
        
        # Extract and print results
        kpss_statistic = result[0]
        p_value = result[1]
        critical_values = result[3]
        
        print('KPSS Statistic:', kpss_statistic)
        print('p-value:', p_value)
        print('Critical Values:')
        for key, value in critical_values.items():
            print(f'{key}: {value}')
        
        # Difference the time series and plot
        diff_data = df.diff().dropna()
        plt.plot(diff_data)
        plt.title('Differenced Time Series')
        plt.show()
        
        # Test for stationarity of the differenced time series using KPSS
        result = kpss(diff_data['IOT_Sensor_Reading'])
        kpss_statistic = result[0]
        p_value = result[1]
        critical_values = result[3]
        
        print('KPSS Statistic:', kpss_statistic)
        print('p-value:', p_value)
        print('Critical Values:')
        for key, value in critical_values.items():
            print(f'{key}: {value}')
    sys.stdout = sys.__stdout__
