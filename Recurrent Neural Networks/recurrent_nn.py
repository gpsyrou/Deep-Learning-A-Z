
# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\hz336yw\\Desktop\\Personal\\Udemy\\Deep_Learning_A_Z\\Deep_Learning_A_Z\\Recurrent_Neural_Networks-1\\Recurrent_Neural_Networks")


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values # Picking only the opening price as a numpy array

