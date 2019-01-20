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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
# As we want to predict price, we have a regression problem and not a classification one
regressor = Sequential()

# Add the first LSTM layer and some Dropout Regularization to avoid overfitting
regressor.add(LSTM(units = 50,return_sequences = True, input_shape =(X_train.shape[1],1)))
regressor.add(Dropout(0.2)) # 20% of the neurons is going to be ignored during each iteration of the training

# Add a second LSTM layer and Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Fourth LSTM layer
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

# Add the output Layer
regressor.add(Dense(units = 1))

# Compile the RNN
# loss = mse as we are having a regression problem
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making the predictions

# Getting the  real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])

X_test= np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
# Finally we have to inverse the transformation so we can get the actual values
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
