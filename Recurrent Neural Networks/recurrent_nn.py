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

# Part 1 - Data Preprocessing

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) # This will bring all stock prices between this range

training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# i.e. at each time t , we will get the previous 60 stock prices
# timesteps: Correspond to the previous 60 financial days (3 previous months)
# 1 output(y): The stock price in t+1

# These list will hold each time the 60 previous stock prices (x) and the output stock price(y)
X_train,y_train = [],[]

# We start from the 60th stock price, so we can retrieve the previous 60
for i in range(60,len(training_set)):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)

# Reshape
# Transforming from 2dim to 3dim
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# 1st dimension corresponds to the number of stock prices
# 2nd dimension corresponds to the number of timesteps
# 3rd dim corresponsd to number of indicators

# Part 2 - Building the RNN








