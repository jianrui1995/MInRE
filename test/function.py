import tensorflow as tf
import numpy as np


def residual_sin(x):
    return x + tf.sin(x)


numpy_x = np.array([1., 2., 3.])
print("function execution in eager mode")
print(type(residual_sin(numpy_x)))