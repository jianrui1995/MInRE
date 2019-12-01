import tensorflow as tf
import preprogram.setting as setting
import gensim

class SentenceLayer():
    def __init__(self):
        self.sentence_dataset = tf.data.TextLineDataset(setting.SENTENCE_PATH)

    def __call__(self):
        dataset = self.sentence_dataset.map(lambda x: tf.py_function(self.sentence2wordvec,inp=[x],Tout=tf.float32))
        return dataset

    def sentence2wordvec(self,c):
        sentence = c.numpy()
        words_list = sentence.decode().split(" ")
        model = gensim.models.Word2Vec.load(setting.WORD2VEC_MODEL_PATH)
        unknown = [0 for _ in range(350)]
        vec = []
        for word in words_list:
            try:
               vec.append(model[word])
            except BaseException:
                vec.append(unknown)
        return [vec]

if __name__ == "__main__":
    s = SentenceLayer()
    dataset = s()
    for data in dataset.take(2):
        print(data)