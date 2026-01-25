import tensorflow as tf

'''     Create a generator with a specified seed
'''

generator_1 = tf.random.Generator.from_seed(42) 

random_1 = generator_1.normal(shape=(2, 2)) # Generate random numbers using the generator



'''     Obtain the global generator
'''

generator_2 = tf.random.get_global_generator()

random_2 = generator_2.normal(shape = (4, 4)) # Generate random numbers using global generator

print(generator_2)

