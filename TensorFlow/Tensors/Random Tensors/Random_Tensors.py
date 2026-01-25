import tensorflow as tf


'''     Created two random (but the same) tensors
'''

random_1 = tf.random.Generator.from_seed(1234) # set seed for reproducibility

random_1 = random_1.normal(shape=(2, 4))

random_2 = tf.random.Generator.from_seed(1234)

random_2 = random_2.normal(shape=(2, 4))


print(random_1)

print(random_1 == random_2)