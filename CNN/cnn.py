import tensorflow as tf
import CNN.setting as setting
import preprogram.setting as Psetting
from preprogram.preprogramoutput import Outputlayer

class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=setting.FILTERS,
            kernel_size=[ setting.KERNELWEIGHT,setting.VEC],
            strides=(1,setting.VEC ),
            padding="VALID",
            data_format="channels_first",
            activation=tf.nn.relu,
            use_bias=True
        )
        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=(Psetting.MAX_WORD_NUM,1),
            padding="SAME",
            data_format="channels_first"
        )
        self.dense = tf.keras.layers.Dense(Psetting.LABEL_NUM,use_bias=False)


    def call(self,input):
        out = self.conv(input)
        out = self.maxpool(out)
        dims = out.shape.dims
        out = tf.reshape(out,[dims[0],-1])
        out = self.dense(out)
        print(out)
        return out


if __name__  == "__main__":
    o = Outputlayer()
    o = o()
    o = o.batch(1).batch(1)
    m = Model()
    for data in o.take(1):
        out = m(data[0])

