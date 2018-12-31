# Part 1 - Importing Data

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

# Step 2 - Preprocessing

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

# Step 3 - Split dataset and Scale

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Each important to perform Feature scaling before we fit a NN
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 4 - Constructing the NN
import keras
# This will initialize our NN
from keras.models import Sequential
# This will craete the Layers of the NN
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input Layer and the first hiddern Layer
# One technique that we can use in order to pick a number of nodes for the hidden
# Layer is to take the average of the number of nodes from the input layer and the output layer
# Thus here 11(numbe rof X's) + 1(output)
# Also we are going to use rectifier for the hidden layer and sigmoid for the output layer
classifier.add(Dense(output_dim = 6, input_dim = 11, 
                     init = 'uniform', activation = 'relu'))

# Adding the second hidden Layer
classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu'))

# Creating the Output Layer
classifier.add(Dense(output_dim = 1 , init = 'uniform', activation = 'sigmoid'))

