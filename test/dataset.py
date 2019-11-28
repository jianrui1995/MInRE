import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'


# dataset = tf.data.Dataset.from_tensor_slices([[8, 3, 0, 8, 2, 1]])
# for ele in dataset:
#     print(ele.numpy())
#
# inc_dataset = tf.data.Dataset.range(100)
# dec_dataset = tf.data.Dataset.range(0, -100, -1)
# dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
# print(dataset)
# batched_dataset = dataset.batch(4)
#
# for batch in batched_dataset.take(4):
#     print([arr.numpy() for arr in batch])
#
# print(batched_dataset)


range_ds  = tf.data.Dataset.range(1000)
#
# batches = range_ds.batch(10,drop_remainder=True)
#
# def dense_1_step(batch):
#     return batch[:-1],batch[1:]
#
# pre = batches.map(map_func=dense_1_step)
# for arr1,arr2 in pre.take(3):
#     print(arr1)
#     print(arr2)

# feature = range_ds.batch(10,drop_remainder=True)
# label = range_ds.batch(10).skip(1).map(lambda batch: batch[:-5])
# for arr in label.take(3):
#     print(arr)
# datasetzip = tf.data.Dataset.zip((feature,label))
# for data in datasetzip.take(3):
#     print(data)

def log_huber(x, m):
  if tf.abs(x) <= m:
    return x**2
  else:
    return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))

x = tf.compat.v1.placeholder(tf.float32)
m = tf.compat.v1.placeholder(tf.float32)

y = tf.py_function(func=log_huber, inp=[x, m], Tout=tf.float32)
dy_dx = tf.gradients(y, x)[0]

with tf.compat.v1.Session() as sess:
  # The session executes `log_huber` eagerly. Given the feed values below,
  # it will take the first branch, so `y` evaluates to 1.0 and
  # `dy_dx` evaluates to 2.0.
  y, dy_dx = sess.run([y, dy_dx], feed_dict={x: 1.0, m: 2.0})

