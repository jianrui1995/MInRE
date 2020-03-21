
from preprogram.preprogramanswer import *
import numpy as np

class Outputlayer():
    def __init__(self,answer_path,sentence_path,sentenceWithentity_path):
        self.sen = VecAndLoc(sentence_path,sentenceWithentity_path)
        self.ans = Answerlayer(answer_path)

    @tf.function
    def __call__(self):
        return tf.data.Dataset.zip((self.sen(),self.ans()))


if __name__ == "__main__":
    o = Outputlayer(*setting.TRAIN_PATH)
    a = o()
    for data in a.padded_batch(4,padded_shapes=([None,None],[None]),padding_values=(0.0,0)).take(1):
        print(data)
