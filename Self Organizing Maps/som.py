# Using Self Organizing maps for Fraud Detection

# Import the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

# Dataset is coming from the UCI: Australian Credit Approval Data set
# Each line of the dataset corresponds to a customer. Thus we will try to find  segments of customers
dataset = pd.read_csv(r'C:\Users\george\Desktop\Online Courses\Udemy\Self_Organizing_Maps\Credit_Card_Applications.csv')
dataset.head()

# For each customer the winning node is the node which is most similar with the customer

# We will split the dataset into two subsets: one containing customers whose application got approved, and another one with customers whose application got declined
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Feature scalling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # scalling so that the values all fall between 0 and 1

X = sc.fit_transform(X)


# Training the Self Organizing Map
import os
os.chdir(r'C:\Users\george\Desktop\Online Courses\Udemy\Self_Organizing_Maps')
from minisom import MiniSom

# sigma = radius
# creating a 10x10 grid, 15 input values
som = MiniSom(x = 10,y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X) # initialize the random weights
som.train_random(data = X,num_iteration = 100)

# Visualizing the results




