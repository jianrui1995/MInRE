import tensorflow as tf
import preprogram.setting as setting

class SentenceLayer():
    def __init__(self):
        self.sentence_dataset = tf.data.TextLineDataset(setting.SENTENCE_PATH)
