import tensorflow as tf
import preprogram.setting as setting

class Answerlayer():
    def __init__(self):
        self.answers2id = {}
        self.answer_dataset = tf.data.TextLineDataset(setting.ANSWER_PATH)

    def __call__(self):
        self.collectanswer(self.answer_dataset)
        dateset = self.answer2id_all()
        return dateset,self.answers2id

    def collectanswer(self,dataset):
        """
        统计数据的分类数量，并建立映射
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
        dataset = self.answer_dataset.map(lambda x: self.convertanswer_map_fun(x))
        return dataset


    def convertanswer(self,t):
        '''
        此方法用于 map_fun 参数
        '''
        id = self.answers2id[t.numpy()]
        return id

    def convertanswer_map_fun(self,t):
        return tf.py_function(self.convertanswer,inp=[t],Tout=(tf.int32))

if __name__ == "__main__":
    an = Answerlayer()
    a,b = an()
    for c in a.take(3):
        print(c)