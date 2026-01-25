import tensorflow as tf

'''     Let's make two tensors 
'''

changable_tensor = tf.Variable([3, 5])

unchangable_tensor = tf.constant([8, 6])

print(changable_tensor)

print(unchangable_tensor)


'''     Let's try to change one of the element in our changable_tensor
'''

# changable_tensor[0] = 2 # Error



'''     Now, we try with .assign()
'''

changable_tensor[0].assign(4)

print(changable_tensor)



'''     Let's try to change our unchangable_tensor
'''

# unchangable_tensor[0].assign(1) # Error: object has no assign attribute


