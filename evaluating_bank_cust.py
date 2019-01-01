
#  Section 6 - Evaluating,Improving and Tuning an ANN

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

# Implementing K-fold cross validation for 10 folds

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, input_dim = 11, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu'))
    classifier.add(Dense(output_dim = 1 , init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics =['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier,batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

# The previous will return 10 different accuracies

mean = accuracies.mean()
variance = accuracies.std()

# Imporving the ANN

# Dropout Regularization to reduce overfitting if needed

# Dropout: At each iteration of the training some neurons are randomly 
# disables , to prevent them be highly dependent when they learn the correlations
# and therefore by overwriting these neurons the ANN learns several independent correlations in the data
# This prevents the neurons from learn too much and thus overfitting

# We can apply dropout in one or several layers
# When we have overfitting its better to apply dropout in all layers

# Creating an ANN with dropouts

classifier = Sequential()

classifier.add(Dense(output_dim = 6, input_dim = 11, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(output_dim = 1 , init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics =['accuracy'])



# Parameter tuning with GridSearch
# GridSearch test multiple combination of hyperparameters and it helps us identify the best combo for our dataset

# Tuning the ANN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, input_dim = 11, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu'))
    classifier.add(Dense(output_dim = 1 , init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer , loss = 'binary_crossentropy' , metrics =['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier)


parameters = {'batch_size':[25,32],
              'epochs': [100,200],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)

'''
{'batch_size': 32, 'epochs': 100, 'optimizer': 'rmsprop'}
0.84525
'''
# Fit the data and train the ANN with the optimal parameters
classifier = build_classifier('rmsprop')
classifier.fit(X_train, y_train, batch_size = 32 , epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print('The accuracy is: ' + str(np.round(((1503 + 222) / len(y_test)) * 100,2)) + '%') 
