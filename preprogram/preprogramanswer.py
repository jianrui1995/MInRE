import tensorflow as tf
import preprogram.setting as setting
# import preprogram.setting_test as setting
from preprogram.preprogramSentence import SentenceLayer
from preprogram.preprogramSentenceWithEntity import SentenceWithEntityLayer

class Answerlayer():
    def __init__(self):
        self.answers2id = setting.answer2id
        self.answer_dataset = tf.data.TextLineDataset(setting.ANSWER_PATH)
        # self.collectanswer(self.answer_dataset)
        print(self.answers2id)
    def __call__(self):
        dateset = self.answer2id_all()
        return dateset

    def collectanswer(self,dataset):
        """
        统计数据的分类数量，并建立映射
        已经丢弃使用，因为不用在统计，在setting文件中直接导入了设置。
        """
        answers = set()
        for answer in dataset:
            answers.add(answer.numpy())
        for v,k in enumerate(answers):
            self.answers2id[k]=v


    def answer2id_all(self):
        '''
        将dataset中所有的answer转换成id
        '''
        dataset = self.answer_dataset.map(lambda x: tf.py_function(self.convertanswer,inp=[x],Tout=tf.int32))
        return dataset


    def convertanswer(self,t):
        '''
        此方法用于 pt_function 方法的参数
        '''
        id = self.answers2id[t.numpy()]
        return id

class VecAndLoc():
    '''
    对词向量和位置向量进行连接
    '''
    def __init__(self):
        self.Sen = SentenceLayer()
        self.SenWEnt = SentenceWithEntityLayer()

    def __call__(self):
        dataset = tf.data.Dataset.zip((self.Sen(),self.SenWEnt()))
        dataset = dataset.map(lambda x,y: self.concat(x,y))
        return dataset

    def concat(self,x,y):
        return tf.concat(values=(x,y),axis=1)



if __name__ == "__main__":
    '''调用Answerlayer'''
    # an = Answerlayer()
    # a = an()
    # b =a.element_spec
    # print(b)
    # for c in a.take(3):
    #     print(c.shape.dims)

    '''调用VecAndLoc'''
    v = VecAndLoc()
    a = v()
    for data in a.take(1):
        print(type(data))
        dims = data.shape.dims
        data = tf.reshape(data,[16,2,-1])
        print(data)
