import tensorflow as tf
import CNN.setting as setting


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filter=setting.FILTERS,
            kerner_size=[setting.VEC, setting.KERNELWEIGHT],
            strides=(setting.VEC, 1),
            padding="VALID",
            data_format="channels_first",
            acitvation=tf.nn.relu,
            use_bias=True
        )

    def call(self,input):
        out = self.conv(input)


if __name__  == "__main__":
    data = tf.constant()
