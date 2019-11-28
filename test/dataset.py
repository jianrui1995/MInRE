import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

dataset = tf.data.Dataset.range(10)
dataset = dataset.shuffle(2).batch(6)
dataset = dataset.repeat(2)
for _ in range(10):
    for arr in dataset:
        print(arr.numpy())