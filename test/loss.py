import tensorflow as tf
print(tf.math.exp(2 * (2.5 - tf.constant(-15.08))))
print(tf.math.exp(2 * (0.5 + tf.constant(46.069016))))
print(tf.math.log(1 + tf.math.exp(2 * (2.5 - tf.constant(-15.08)))) + tf.math.log(1 + tf.math.exp(2 * (0.5 + tf.constant(46.069016)))))
