import tensorflow as tf
import preprogram.setting as setting
import re
import numpy as np


class SentenceWithEntityLayer():
    def __init__(self):
        self.sentence_entity_dataset = tf.data.TextLineDataset(setting.SENTENCE_WITH_ENTITY_PATH)
        self.loc_dic = { i:np.random.random(50) for i in range(-1*setting.MAX_WORD_NUM,setting.MAX_WORD_NUM)}

    def __call__(self):
        dataset = self.sentence_entity_dataset.map(lambda x: tf.py_function(self.entity2location,inp=[x],Tout=tf.float64))
        return dataset

    def entity2location(self,t):
        # global location1, location2
        words_list = t.numpy().decode().split(" ")
        mark1,mark2 = 1,1
        loc_list = []
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
        for i in range(len(words_list)):
            loc1 = self.loc_dic[location1-i]
            loc2 = self.loc_dic[location2-i]
            loc_list.append(np.append(loc1,loc2))
        return [loc_list]

if __name__ == "__main__":
    S = SentenceWithEntityLayer()
    dataset = S()
    print(dataset.element_spec)
    for s in dataset.take(1):
        # print(s)
        pass
    # f = open(setting.SENTENCE_WITH_ENTITY_PATH,"r",encoding="utf8")
    # for data in f.readlines():
    #     print(data)