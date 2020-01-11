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
        self.dense = tf.keras.layers.Dense(Psetting.LABEL_NUM,use_bias=False)

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

class Train_Test():
    def __init__(self,path=True):
        self.model = Model()
        self.outputobj = Outputlayer()
        self.output = self.outputobj()
        # 载入模型
        if path != True:
            self.model.load_weights(path)


    def train(self):
        output = self.output.batch(1).batch(1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        for _ in range(setting.EPOCH):
            print(_)
            for data in output:
                with tf.GradientTape() as tape:
                    out = self.model(data[0])
                    loss = self.loss(out,data[1])
                grads = tape.gradient(loss,self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
            break
        """输入模型保存的位置"""
        self.model.save_weights("../model/ModelWeights.ckpt")


    def test(self):
        '''
        测试用的方法
        '''
        f = open("result.txt","w",encoding="utf8")
        output = self.output.batch(1).batch(1)
        for data in output:
            out = self.model(data[0])
            _,num = tf.math.top_k(out,1)
            print(num.numpy(),file=f,end="\n")
        f.close()


    # @tf.function
    def loss(self,input,label):
        '''
        损失函数
        '''
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
    # t = Train_Test()
    # t.train()

    """输入模型载入的路径"""
    t = Train_Test("../model/ModelWeights.ckpt")
    t.test()
