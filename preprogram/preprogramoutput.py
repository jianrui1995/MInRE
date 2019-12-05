import tensorflow as tf
from preprogram.preprogramanswer import *

class Outputlayer():
    def __init__(self):
        self.sen = VecAndLoc()
        self.ans = Answerlayer()

    def __call__(self):
        return tf.data.Dataset.zip((self.sen(),self.ans()))


if __name__ == "__main__":
    o = Outputlayer()
    a = o()
    for data in a.take(1):
        print(data)