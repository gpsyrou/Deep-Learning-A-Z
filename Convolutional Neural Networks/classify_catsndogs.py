
# Convolutional Neural Networks (CNN)

# Task: Create a CNN that takes as input images of dogs and cats
#       and predicts in which category each picture belongs

# Building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D # 2D as we are using images. For videos we need the 3D
from keras.layers import MaxPool2D # For the pooling layer
from keras.layers import Flatten # For the flattening step
from keras.layers import Dense # Used to add the fully connected layer

# Initialize the CNN
classifier = Sequential()

# Add the Layers of the CNN

# Step 1 - Convolution : Applying several feature detectors 
# to the input image and create a feature map for each feature detector
# This step creates many feature maps to obtain each convolutional layer

# 32 =  Number of feauture maps constructed from 3x3 feature detectors
# 3,3 = dimension of each feature detector

classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))

# Step 2 - Pooling Layer
# Used to reduce the size of the feature maps
# i.e. it creates a smaller feature map
# It is used to reduce the number of nodes in the future fully connected layer
classifier.add(MaxPool2D(pool_size = (2,2)))

# Step 3 - Flattening
# We are flatteting all the Pooling Layers into one huge vector that will be the input to the ANN
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(output_dim = 128 , activation = 'relu' ))
# sigmoid as we have binary outcome/ if we had more than 2 then we use softmax activation function
classifier.add(Dense(output_dim = 1 , activation = 'sigmoid' )) 

#Step 5 - Initialise the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])




