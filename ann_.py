#  Section 4 - Building an Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Goal: Predict which costumers are going to leave the bank (binary classification)

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# First check which variables seem useful to predict which costumers are going to leave the bank
# The first few columns dont provide any useful

X = dataset.iloc[:, 3:13].values # contains thwe useful features
y = dataset.iloc[:, 13].values # contains the binary outcomes
