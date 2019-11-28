import tensorflow as tf
import preprogram.setting as setting

class Answerlayer(tf.keras.layers.Layer):
    def __init__(self):
        super(Answerlayer,self).__init__()
        '''记得修改这里'''
        self.answer_dataset = tf.data.TextLineDataset(setting.ANSWER_PATH)

    def call(self, inputs, **kwargs):
        pass


    def collectanswer(self,dataset):
        """
        统计数据的分类数量，并建立映射
        """
        answers2id = {}
        answers = set()
        for answer in dataset.take(4):
            answers.add(answer.numpy())
        for v,k in enumerate(answers):
            answers2id[k]=v
        return answers2id

    def answer2id_all(self,dataset,answer2id):
        '''
        将dataset中所有的answer转换成id
        '''
        for ar in dataset.take(2):
            print(ar)
        dataset.map(lambda x: self.convertanswer(x,answer2id))
        for a in dataset.take(3):
            print(a)

    def convertanswer(self,t,answers2id):
        '''
        此方法用于 map_fun 参数
        '''
        print(type(t))
        id = answers2id[t.numpy()]
        return t

if __name__ == "__main__":
    an = Answerlayer()
    ans2id = an.collectanswer(an.answer_dataset) #这个是拆分了的
    an.answer2id_all(an.answer_dataset,ans2id)