import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

class Setting():
    def __init__(self):
        # 设置输出通道数量
        self.FILTERS = 1000
        # 设置词向量维度
        self.WORDVEC = 350
        # 设置位置向量维度
        self.POSVEC = 50
        # 总体向量维度
        self.VEC = self.POSVEC + self.WORDVEC
        # 卷积核宽
        self.KERNELWEIGHT = 4

class model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.setting = Setting()
        self.conv = tf.keras.layers.Conv2D(
            filter=self.setting.FILTERS,
            kerner_size=[self.setting.VEC, self.setting.KERNELWEIGHT],
            strides=(1, 1),
            padding="SAME",
            data_format="channels_first",
            acitvation=tf.nn.relu,
            use_bias=True
        )

    def call(self,input):
        out = self.conv(input)


if __name__  == "__main__":
    data = tf.constant()
