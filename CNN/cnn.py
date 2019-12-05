import tensorflow as tf
import CNN.setting as setting
import preprogram.setting as Psetting
from preprogram.preprogramoutput import Outputlayer

class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            filter=setting.FILTERS,
            kerner_size=[setting.VEC, setting.KERNELWEIGHT],
            strides=(setting.VEC, 1),
            padding="VALID",
            data_format="channels_first",
            acitvation=tf.nn.relu,
            use_bias=True
        )
        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=(1,Psetting.MAX_WORD_NUM),
            padding="SAME",
            data_format="channels_first"
        )
        self.dense = tf.keras.layers.Dense(Psetting.LABEL_NUM,use_bias=False)
    def call(self,input):
        out = self.conv(input)
        out = self.maxpool(out)
        dims = out.shape.dims
        out = tf.reshape(out,[dims[0],-1])
        return out
if __name__  == "__main__":
    o = Outputlayer()
    o = o()
    o = o.batch(2)
    for data in o.take(1):
        print(data)
