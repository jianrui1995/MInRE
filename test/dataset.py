import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

dataset = tf.data.Dataset.from_tensor_slices([[8, 3, 0, 8, 2, 1]])
for ele in dataset:
    print(ele.numpy())

inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
print(dataset)
batched_dataset = dataset.batch(4)

for batch in batched_dataset.take(4):
    print([arr.numpy() for arr in batch])

print(batched_dataset)
