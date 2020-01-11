import tensorflow as tf
import preprogram.setting as setting
# import preprogram.setting_test as setting
import gensim

class SentenceLayer():
    def __init__(self):
        self.sentence_dataset = tf.data.TextLineDataset(setting.SENTENCE_PATH)
        self.model = gensim.models.Word2Vec.load(setting.WORD2VEC_MODEL_PATH)

    def __call__(self):
        dataset = self.sentence_dataset.map(lambda x: tf.py_function(self.sentence2wordvec,inp=[x],Tout=tf.float64))
        return dataset

    def sentence2wordvec(self,c):
        sentence = c.numpy()
        words_list = sentence.decode().split(" ")
        unknown = [0 for _ in range(350)]
        vec = []
        for word in words_list:
            try:
               vec.append(self.model[word])
            except BaseException:
                vec.append(unknown)
        return [vec]

if __name__ == "__main__":
    s = SentenceLayer()
    dataset = s()
    print(dataset.element_spec)
    for data in dataset.take(1):
        print(data)