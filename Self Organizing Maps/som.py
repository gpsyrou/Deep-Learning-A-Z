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
