import tensorflow as tf
import preprogram.setting as setting
from preprogram.preprogramSentence import SentenceLayer
from preprogram.preprogramSentenceWithEntity import SentenceWithEntityLayer

class Answerlayer():
    def __init__(self,answer_path):
        self.answers2id = setting.answer2id
        self.answer_dataset = tf.data.TextLineDataset(answer_path)


    def __call__(self):
        dateset = self.answer_dataset.map(lambda x: tf.py_function(func=self.answer2id_all,inp=[x],Tout=tf.int32))
        return dateset



    def answer2id_all(self,input):
        '''
        将dataset中所有的answer转换成id
        '''
        label = [0 for _ in range(setting.LABEL_NUM)]
        label[self.answers2id[input.numpy()]] = 1
        return [label]


class VecAndLoc():
    '''
    对词向量和位置向量进行连接
    '''
    def __init__(self,sentence_path,sentenceWithentity_path):
        self.Sen = SentenceLayer(sentence_path)
        self.SenWEnt = SentenceWithEntityLayer(sentenceWithentity_path)

    def __call__(self):
        dataset = tf.data.Dataset.zip((self.Sen(),self.SenWEnt()))
        dataset = dataset.map(lambda x,y: self.concat(x,y))
        return dataset

    def concat(self,x,y):
        return tf.concat(values=(x,y),axis=1)



if __name__ == "__main__":
    '''调用Answerlayer'''
    an = Answerlayer()
    a = an()
    for c in a.take(3):
        print(c)

    '''调用VecAndLoc'''
    # v = VecAndLoc()
    # a = v()
    # for data in a.take(1):
    #     print(data)
