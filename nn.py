import os

import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

# flatten the input

x_train = x_train.reshape(-1, 28*28).astype('float32')/255.0

x_test = x_test.reshape(-1, 28*28).astype('float32')/255.0

# if the data is already in numpy array, it will be converted to tensor automatically

# Create a neural network from sequential API 

# set the structure for the model 
model = keras.Sequential(
    [
        layers.Dense(512, activation = 'relu'),
        layers.Dense(256, activation = 'relu'),
        layers.Dense(10),
    ]
)


# configure/compile the model 
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer= keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy']
)

# fit the model
hyper_param = {
    'batch_size': 32, 
    'epochs': 5, 
    'verbose': 2,
}



model.fit(x_train, y_train, batch_size = 32, epochs= 5, verbose=2)



