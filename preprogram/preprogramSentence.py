import tensorflow as tf
import preprogram.setting as setting
import gensim

class SentenceLayer():
    def __init__(self,sentence_path):
        self.sentence_dataset = tf.data.TextLineDataset(sentence_path)
        self.model = gensim.models.KeyedVectors.load_word2vec_format(setting.GLOVE_MODEL_PATH)

    def __call__(self):
        dataset = self.sentence_dataset.map(lambda x: tf.py_function(self.sentence2wordvec,inp=[x],Tout=tf.float32))
        return dataset

    def sentence2wordvec(self,c):
        sentence = c.numpy()
        words_list = sentence.decode().split(" ")
        vec = []
        for word in words_list:
            try:
               vec.append(self.model[word])
            except BaseException:
                vec.append(self.model["<unk>"])
        return [vec]

if __name__ == "__main__":
    s = SentenceLayer()
    dataset = s()
    for data in dataset.take(1):
        print(data)