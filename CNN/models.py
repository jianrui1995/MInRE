# @Time: 2020/3/10 20:23
# @Author: R.Jian
# @Note: CNN关系分类模型。
import sys
sys.path.append(r"/home/tech/myPthonProject/MinRE/")
import tensorflow as tf

import CNN.setting as setting
import CNN.layers as layers
from preprogram.preprogramoutput import Outputlayer
import CNN.metrics as metrics
import CNN.callbacks as callbacks

class CNN(tf.keras.models.Model):
    def __init__(self):
        super(CNN,self).__init__()

        self.transshape = layers.TransShape()

        self.cov2d = tf.keras.layers.Conv2D(
            filters=setting.FILTERS,
            kernel_size=[setting.KERNELWEIGHT,setting.VEC],
            strides=(1,setting.VEC),
            padding="SAME",
            data_format="channels_last",
            activation=tf.keras.activations.relu
        )

        "max层，但是还需要计算mask的范围"

        # self.maxpool = tf.keras.layers.MaxPool2D(
        #     pool_size=()
        # )
        self.dense_1 = tf.keras.layers.Dense(
            units=setting.DENSE_1_NUIM,
            activation=tf.keras.activations.relu
        )

        self.dense_2 = tf.keras.layers.Dense(
            units=setting.DENSE_2_NUIM,
            activation=tf.keras.activations.relu
        )

        self.dense_3 = tf.keras.layers.Dense(
            units=setting.DENSE_3_NUIM,
            activation=tf.keras.activations.softmax
        )

    @tf.function(input_signature=(tf.TensorSpec(shape=[None,None,None]),))
    def call(self, inputs):
        outputs = self.transshape(inputs,typemodel="A")
        outputs = self.cov2d(outputs)
        outputs = self.transshape(outputs,typemodel="B")
        outputs = self.dense_1(outputs)
        outputs = self.dense_2(outputs)
        outputs = self.dense_3(outputs)
        return outputs

if __name__ == "__main__":
    trainset = Outputlayer(*setting.TRAIN_PATH)().padded_batch(4,padded_shapes=([None,None],[None]),padding_values=(0.0,0))
    testSet = Outputlayer(*setting.TEST_PATH)().padded_batch(4,padded_shapes=([None,None],[None]),padding_values=(0.0,0))
    model = CNN()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[metrics.Precison()]
    )
    model.fit(
        x=trainset,
        epochs=setting.EPOCH,
        validation_data=testSet,
        validation_freq=setting.SAVE_N_EPOCH,
        callbacks=[callbacks.Save(setting.SAVE_N_EPOCH,setting.SAVE_PATH,setting.SAVE_NAME)],
    )
