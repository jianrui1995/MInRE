import tensorflow as tf
import CNN.setting as setting
import preprogram.setting as Psetting
from preprogram.preprogramoutput import Outputlayer

class Model(tf.keras.Model):
    def __init__(self):
        init = tf.keras.initializers.RandomNormal()
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
            data_format="channels_first",
        )
        self.dense = tf.keras.layers.Dense(Psetting.LABEL_NUM,use_bias=False,activation=tf.keras.activations.softmax)

    # @tf.function
    def call(self,input):
        out = self.conv(input)
        # dims = out.shape.dims
        # print(dims)
        out = self.maxpool(out)
        dims = out.shape.dims
        # print(dims)
        out = tf.reshape(out,[dims[0],-1])
        out = self.dense(out)
        dims = out.shape.dims
        # print(dims)
        # print(out)
        # out = tf.keras.activations.tanh(out)
        # out = tf.matmul(tf.constant([[20.0]],dtype=tf.float32),out)
        return out

class Train():
    def __init__(self):
        self.outputobj = Outputlayer()
        self.output = self.outputobj()
        self.model = Model()

    def train(self):
        f1 = open("out.txt","w",encoding="utf8")
        f2 = open("loss.txt","w",encoding="utf8")
        f3 = open("grads.txt","w",encoding="utf8")
        output = self.output.batch(1).batch(1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # 观察数据：在model输出计算梯度和加上loss函数计算梯度。比较两个导数的差异。
        for _ in range(setting.EPOCH):
            for data in output:
                with tf.GradientTape() as tape:
                    out = self.model(data[0])
                    print("out:",out,file=f1)
                    loss = self.loss(out,data[1])
                    print("loss:",loss,file=f2)
                grads = tape.gradient(loss,self.model.trainable_variables)
                print("grads:",grads,file=f3)
                optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
            break
    @tf.function
    def loss(self,input,label):
        # print(label.numpy())
        labels = tf.sparse.SparseTensor(indices=[[0,label.numpy()[0][0]]],values=[1],dense_shape=[1,Psetting.LABEL_NUM])
        return tf.nn.softmax_cross_entropy_with_logits(tf.sparse.to_dense(labels),input)


    """
    被丢弃的，论文中用的损失函数。无法解决指数超差计数范围的问题。
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
                # return 1
            else:
                # 最大值不为对应的label
                return tf.math.log(1+tf.math.exp(setting.R*(setting.M_NEG+tf.gather(order[0],0,axis=1))))
                # return 2
        else:
            order = tf.math.top_k(out,2)
            if label == tf.gather(order[1],0,axis=1):
                # 若第一个就是label，则取第二大值
                return tf.math.log(1+tf.math.exp(setting.R*((setting.M_POS-tf.gather(tf.reshape(out,[-1]),tf.reshape(label,[-1]))))))+tf.math.log(1+tf.math.exp(setting.R*(setting.M_NEG+tf.gather(order[0],1,axis=1))))
                # return 3
            else:
                # 最大值不为对应的label
                return tf.math.log(1+tf.math.exp(setting.R*((setting.M_POS-tf.gather(tf.reshape(out,[-1]),tf.reshape(label,[-1]))))))+tf.math.log(1+tf.math.exp(setting.R*(setting.M_NEG+tf.gather(order[0],0,axis=1))))
                # return 4
    """

if __name__  == "__main__":
    t = Train()
    t.train()
