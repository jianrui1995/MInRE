import tensorflow as tf
import numpy as np

a = [[0,1,2,3,4],[1,2,3]]
a = np.array(a)
print(a)
A = tf.data.Dataset.from_generator(lambda: iter(a), tf.int32)
for data in A:
    print(data)
