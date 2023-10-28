import pandas as pd
import numpy as np
import sys 
from datetime import datetime
from projectpro import checkpoint
from ML_Pipeline import Preprocess
from ML_Pipeline import EDA
from ML_Pipeline import Stationarity
from ML_Pipeline import acf
from ML_Pipeline import pacf
from ML_Pipeline import WhiteNoise 
from ML_Pipeline import RandomWalk
from ML_Pipeline import MA_model
from ML_Pipeline import Autoregressor

def main():
    # Importing the data
    raw_csv_data = pd.read_csv('../input/' + 'Data-Chillers.csv') 

    df = raw_csv_data.copy()  # Create a copy of the data
    checkpoint('c2bf09')  # Create a checkpoint for the project

    # Preprocess the data
    df = Preprocess.preprocess(df) 
    
    # Calculate the covariance matrix
    ac_data = df.set_index('time')
    
    with open('../output/' + 'log.txt', 'a') as f:
        sys.stdout = f
        print("Run time:", datetime.now())
        print("**************** Autocovariance matrix value is: ***********************")
        print(np.cov(ac_data, rowvar=False))
    sys.stdout = sys.__stdout__  # Restore the standard output

    # Exploratory Data Analysis (EDA)
    components_plot = EDA.EDA(df.copy())
    
    # Check for stationarity
    Stationarity.Stationary(df.copy())
   
    # Calculate and plot autocorrelation function (ACF)
    acf_plot = acf.acf(df.copy())
    
    # Calculate and plot partial autocorrelation function (PACF)
    pacf_plot = pacf.pacf(df.copy())
    
    # Check for white noise
    whitenoise_plot = WhiteNoise.white_noise(df.copy())
    checkpoint('c2bf09')  # Create another checkpoint

    # Generate and plot a random walk
    randomwalk_plot = RandomWalk.random_walk(df.copy())
    
    # Fit and summarize a Moving Average (MA) model
    MA_summary = MA_model.MA_model(df.copy())
    
    # Fit and summarize a First-order AutoRegressive (AR) model
    AR_summary = Autoregressor.AR_model(df.copy())
    
    return components_plot, acf_plot, pacf_plot, randomwalk_plot, whitenoise_plot, randomwalk_plot, MA_summary, AR_summary

main()
