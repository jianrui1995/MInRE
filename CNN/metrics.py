# @Time: 2020/3/18 18:12
# @Author: R.Jian
# @Note: 评价对象的方法

import tensorflow as tf

class Precison(tf.keras.metrics.Metric):
    def __init__(self):
        super(Precison,self).__init__(name="precision",dtype=tf.float32)
        self.total = self.add_weight(name="total",shape=(),initializer=tf.zeros,dtype=tf.float32)
        self.correct = self.add_weight(name="correct",shape=(),initializer=tf.zeros,dtype=tf.float32)

    def update_state(self,y_true,y_pre, *args, **kwargs):
        self.total = self.total.assign_add(tf.cast(tf.shape(y_true,out_type=tf.int64)[0],tf.float32))
        max = tf.math.argmax(y_pre,axis=-1,output_type=tf.int64)
        max = tf.reshape(max,shape=[-1,1])
        rank = tf.range(0,tf.shape(y_pre,out_type=tf.int64)[0],1,dtype=tf.int64)
        rank = tf.reshape(rank,[-1,1])
        pre = tf.sparse.SparseTensor(tf.concat([rank,max],axis=-1),
                                     tf.ones(shape=tf.shape(y_pre,out_type=tf.int64)[0],dtype=tf.int64),
                                     tf.shape(y_pre,out_type=tf.int64))
        pre = tf.sparse.to_dense(pre)
        result = tf.math.reduce_sum(tf.math.multiply(pre,tf.cast(y_true,dtype=tf.int64)))
        self.correct.assign_add(tf.cast(result,dtype=tf.float32))


    def result(self):
        return tf.math.divide(self.correct,self.total)

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
        tf.keras.callbacks.History


