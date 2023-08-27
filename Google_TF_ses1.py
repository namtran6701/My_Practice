import tensorflow as tf

# constant variable 
x = tf.constant([
    [3,5,7],
    [4,6,8]
]
)
  
# We can also reshape the matrix 

y = tf.reshape(x, [3, 2])

y 

# tensor-variable 

x = tf.Variable(2, dtype = tf.float32, name = 'my_variable')

x.assign(67)

# add value to the existing value 

x.assign_add(3)

# subtract the value of the current value 

x.assign_sub(10)