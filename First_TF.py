import tensorflow as tf
from tensorflow import keras
import numpy as np

# Set up x and y

x = np.array([-1, 0, 1, 2, 3, 4], dtype=float)

y = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

# Create the model structure 
model = keras.Sequential(
    [keras.layers.Dense(units = 1, input_shape = [1])]
)

# Compile the model
'''
When compiling the model, we define the optimizer, loss function and metrics. 
- optimizer: update the model weights based on loss function
- loss fucntion: this is what the model try to minimize during training 
- metrics: used to judge the model performance: accuracy, precision, recall. 
'''
model.compile(optimizer='sgd', loss='mean_squared_error')

# fit the model 

model.fit(x, y, epochs = 500)


# Predict new value 

print(model.predict([10.0]))
