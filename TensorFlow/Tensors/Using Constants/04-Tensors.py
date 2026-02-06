import tensorflow as tf

tensor = tf.constant([
    [ 
        [2, 3, 4],
    ],
    [
        [6, 7, 5],
    ]
])

'''
shape = (2, 1, 3)

'''


print(tensor)

'''     Let's check it's dimensions
'''

print(tensor.ndim)