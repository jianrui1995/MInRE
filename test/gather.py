import tensorflow as tf

input = tf.constant([[5,4,3,0,1],[2,3,0,4,2],[2,3,5,4,2]])
input = input + 1
print(tf.reshape(tf.gather_nd(input,[[1,2]]),[-1]))