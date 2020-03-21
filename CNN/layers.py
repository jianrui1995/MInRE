# @Time: 2020/3/16 17:27
# @Author: R.Jian
# @Note: 自定义的相关层

import tensorflow as tf

class TransShape(tf.keras.layers.Layer):
    def __init__(self):
        super(TransShape,self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs, typemodel, training=None):
        "typemodel确定tensor的转换模式，模式不同转化的方式也不同，主要是由于都是转换，就融合在一起写了"
        if typemodel == "A":
            tensorshape = inputs.shape
            outputs = tf.reshape(inputs,[tensorshape[0],-1,tensorshape[1],tensorshape[2]])
            outputs = tf.transpose(outputs,[0,2,3,1])
        if typemodel == "B":
            tensorshape = inputs.shape
            outputs = tf.transpose(inputs,[0,3,1,2])
            outputs = tf.reshape(outputs,[tensorshape[0],tensorshape[3],-1])
            outputs = tf.reduce_max(outputs,axis=-1)
        return outputs

