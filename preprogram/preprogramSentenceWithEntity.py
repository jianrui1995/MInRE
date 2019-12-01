import tensorflow as tf
import preprogram.setting as setting
import re

class SentenceWithEntityLayer():
    def __init__(self):
        self.sentence_entity_dataset = tf.data.TextLineDataset(setting.SENTENCE_WITH_ENTITY_PATH)

    def __call__(self):
        dataset = self.sentence_entity_dataset.map(lambda x: tf.py_function(self.entity2location,inp=[x],Tout=[tf.int32]))
        return dataset

    def entity2location(self,t):
        print(t.numpy())
        words_list = t.numpy().decode().split(" ")
        mark1,mark2 = 1,1
        for location, word in enumerate(words_list):
            if mark1:
                if re.match(r"<e1>",word):
                    location1 = location
                    mark1 = 0
            if mark2:
                if re.match(r"<e2>",word):
                    location2 = location
                    mark2 = 0
            if ( not mark1) and ( not mark2):
                break
        return [[location1,location2]]

if __name__ == "__main__":
    S = SentenceWithEntityLayer()
    dataset = S()
    for s in dataset.take(1):
        print(s)

    # f = open(setting.SENTENCE_WITH_ENTITY_PATH,"r",encoding="utf8")
    # for data in f.readlines():
    #     print(data)