import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

# Create the model structure
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)), # Flatten the 28x28 image to 1x784
    keras.layers.Dense(128, activation = tf.nn.relu), # 128 neurons, activation function: relu
    keras.layers.Dense(10, activation = tf.nn.softmax) # 10 neurons, activation function: softmax
])

# compile the model 
model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

# fit the model 
model.fit(train_images, train_labels, epochs = 5)

# evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

