import torch 

# Let's explore a few basic tensor manipulation

# Create a 5x3 matrix filled with zeros. 
z = torch.zeros(5, 3)
print(z)

print(z.dtype)

# As we can see, the default dtype is float32. We can override this by specifying the dtype

z = torch.ones(5,3, dtype = torch.int16)

print(z.dtype)

# Create a matrix with random number

torch.manual_seed(0)
z = torch.rand(5,3)

print(z)
