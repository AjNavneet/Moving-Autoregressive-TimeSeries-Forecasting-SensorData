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

# Create a class for the random walk
def random_walk(df):
    # Set random seed for reproducibility

    # Parameters
    num_steps = 1000
    initial_value = 0
    
    # Generate random steps
    steps = np.random.choice([-1, 1], size=num_steps)
    
    # Accumulate steps to simulate the random walk
    random_walk = np.cumsum(steps) + initial_value
    
    # Plot the random walk
    plt.plot(random_walk)
    plt.title('Random Walk')
    plt.xlabel('Step')
    plt.ylabel('Value')
    
    # Save the plot as 'random_walk.png' in the '../output/' directory
    plt.savefig('../output/' + 'random_walk.png')
    
    # Display the plot
    return plt.show()
