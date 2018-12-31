#  Section 4 - Building an Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries

# Data processing
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt



# Goal: Predict which costumers are going to leave the bank (binary classification)

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# First check which variables seem useful to predict which costumers are going to leave the bank
# The first few columns dont provide any useful

X = dataset.iloc[:, 3:13].values # contains thwe useful features
y = dataset.iloc[:, 13].values # contains the binary outcomes

# We have some categorical variables, so at first point we need to 
# encode the categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label_encoder_1 = LabelEncoder()
X[:,1] = label_encoder_1.fit_transform(X[:,1]) # Encoding Countries

label_encoder_2 = LabelEncoder()
X[:,2] = label_encoder_2.fit_transform(X[:,2]) # Encoding Sex

# Create dummy variables for Countries
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
