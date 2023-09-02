import os 
import tensorflow as tf

# Initialization of tensors 
x = tf.constant(4, shape = (1,1), dtype = tf.float32)

# A matrix full of ones with specified shape
x = tf.ones(shape = (3, 3), dtype = tf.float32)
print(x)

# A matrix full of zeros with specified shape
x = tf.zeros(shape = (3,4), dtype = tf.float32)
print(x)

# a matrix full of zeros with 1 laid in the diag line, provide the total number of ones 
x = tf.eye(4)
print(x)

# Mathematical operations 
x = tf.constant([1,2,3])
y = tf.constant([7,8,9])

# Addition
z = tf.add(x,y)
#or
z = x+y
print(z)

# Multiplication
z = tf.mutiply(x,y)
#or
z = x*y
print(z)

# Dot product between two vector
z = tf.tensordot(x, y, axes = 1)
print(z)

# Matrix multiplication 
x = tf.random.normal(shape = (2,3))
y = tf.random.normal(shape = (3,4))

z = tf.matmul(x, y)

## or we can do as follow 
x@y

# indexing matrix in tensorflow 
x = tf.constant([0,6,7,1,2,3,4])

print(x[1:])
print(x[::2])
 

# Specify indices of a matrix 
indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)

# Slice a row a column from a matrix 

x = tf.constant([
    [1,2],
    [3,4],
    [5,6]
])

print(x[0,:])

print(x[:, 1])

print(x[0:2,:])

# Reshape a matrix

x = tf.range(9)

z = tf.reshape(x, shape = (3,3))
print(z)

# transpose a matrix 
tf.transpose(z, perm = [1,0])

''' 
by setting perm = [1,0], we convert column to row
'''
