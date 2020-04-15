# @Time: 2020/3/16 17:27
# @Author: R.Jian
# @Note: 自定义的相关层

import tensorflow as tf
import CNN.setting as setting
class TransShape(tf.keras.layers.Layer):
    def __init__(self):
        super(TransShape,self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs, typemodel, training=None):
        "typemodel确定tensor的转换模式，模式不同转化的方式也不同，主要是由于都是转换，就融合在一起写了"
        print()
        print(inputs)
        if typemodel == "A":
            print("A")
            tensorshape = tf.shape(inputs)
            print(tensorshape)
            outputs = tf.reshape(inputs,[tensorshape[0],1,tensorshape[1],tensorshape[2]])
            outputs = tf.transpose(outputs,[0,2,3,1])
            print(outputs)
        if typemodel == "B":
            print("B")
            tensorshape = tf.shape(inputs)
            outputs = tf.transpose(inputs,[0,3,1,2])
            print(outputs) #(None, 1000, None, None)
            outputs = tf.reshape(outputs,[tensorshape[0],tensorshape[3],tensorshape[1]])
            print(outputs)
            outputs = tf.reduce_max(outputs,axis=-1)
            outputs.set_shape([None,setting.FILTERS])
            print(outputs)
        return outputs

