import tensorflow as tf

choose = tf.constant([[5,4,3,0,1],[2,3,0,4,2],[2,3,5,4,2]])
sort = tf.nn.top_k(choose,2)
print(sort[0],sort[1])