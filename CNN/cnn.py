import tensorflow as tf
import CNN.setting as setting
import preprogram.setting as Psetting
from preprogram.preprogramoutput import Outputlayer

class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=setting.FILTERS,
            kernel_size=[setting.KERNELWEIGHT,setting.VEC],
            strides=(1,setting.VEC ),
            padding="SAME",
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

    # @tf.function
    def call(self,input):
        out = self.conv(input)
        out = self.maxpool(out)
        dims = out.shape.dims
        out = tf.reshape(out,[dims[0],-1])
        print("out",out.shape)
        out = self.dense(out)
        return out

class Train():
    def __init__(self):
        self.outputobj = Outputlayer()
        self.output = self.outputobj()
        self.model = Model()

    def train(self):
        output = self.output.batch(1).batch(1)
        for _ in range(setting.EPOCH):
            for data in output:
                out = self.model(data[0])
                self.model.variables
                self.model.trainable_variables
                loss = self.loss(out,data[1],self.outputobj.ans.answers2id)
                print(loss,data[1])
                print(data[0].shape)
            break

    @tf.function
    def loss(self,out,label,label_dict):
        '''
        计算损失函数
        '''
        if label_dict["Other".encode()] == label:
            order = tf.math.top_k(out,2)
            if label == tf.gather(order[1],0,axis=1):
                # 若第一个就是最大值，则取第二大值
                return tf.math.log(1+tf.math.exp(setting.R*(setting.M_NEG+tf.gather(order[0],1,axis=1))))
            else:
                # 最大值不为对应的label
                return tf.math.log(1+tf.math.exp(setting.R*(setting.M_NEG+tf.gather(order[0],0,axis=1))))
        else:
            order = tf.math.top_k(out,2)
            if label ==tf.gather(order[1],0,axis=1):
                # 若第一个就是label，则取第二大值
                return tf.math.log(1+tf.math.exp(setting.R*((setting.M_POS-tf.gather(out,label,axis=1)))))+tf.math.log(1+tf.math.exp(setting.R*(setting.M_NEG+tf.gather(order[0],1,axis=1))))
            else:
                # 最大值不为对应的label
                return tf.math.log(1+tf.math.exp(setting.R*((setting.M_POS-tf.gather(tf.reshape(out,[-1]),tf.reshape(label,[-1]))))))+tf.math.log(1+tf.math.exp(setting.R*(setting.M_NEG+tf.gather(order[0],0,axis=1))))

if __name__  == "__main__":
    t = Train()
    t.train()
