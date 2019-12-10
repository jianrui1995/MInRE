import tensorflow as tf

input = tf.constant([5,4,3,0,1])
print(tf.gather(input,[1],axis=0))