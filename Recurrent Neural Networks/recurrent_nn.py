# Task: Predict the trend of Googles stock price for the year 2017
# As a training set we are using the stock prices from 2012 to 2016

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\hz336yw\\Desktop\\Personal\\Udemy\\Deep_Learning_A_Z\\Deep_Learning_A_Z\\Recurrent_Neural_Networks-1\\Recurrent_Neural_Networks")


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values # Picking only the opening price as a numpy array

# Whenever we are building an RNN then Normalisation is recommended, instead of Standardization

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) # This will bring all stock prices between this range


training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# timesteps: Correspond to the previous 60 financial days (3 previous months)
# 1 output: The stock price