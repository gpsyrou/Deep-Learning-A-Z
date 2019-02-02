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
