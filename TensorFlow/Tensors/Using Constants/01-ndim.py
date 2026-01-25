import tensorflow as tf

'''     Checking the dimensions (ndim stands for number of dimensions)
'''


'''     Checking scalar dimensions
'''

scalar = tf.constant(6)

print(scalar.ndim) # 0


'''     Now try on vectors
'''

vector = tf.constant([2, 4])

print(vector.ndim) # 1



'''     Now try with matrix
'''

matrix = tf.constant([
        [4, 5],
        [6, 7]
])

print(matrix.ndim) # 2