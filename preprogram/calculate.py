import preprogram.setting as setting
import tensorflow as tf
from preprogram.preprogramanswer import Answerlayer
from preprogram.preprogramSentence import SentenceLayer
from preprogram.preprogramSentenceWithEntity import SentenceWithEntityLayer

'''计算最大的句子长度'''
# max = 0
# with open(setting.SENTENCE_PATH,"r",encoding="utf8") as f:
#     for data in f.readlines():
#         word_list = data.strip().split(" ")
#         if max < len(word_list):
#             max = len(word_list)
# print( max)

'''测试dataset.zip()'''
ans = Answerlayer()
Sen = SentenceLayer()
SenWEnt = SentenceWithEntityLayer()
print(ans().element_spec)
print(Sen().element_spec)
print(SenWEnt().element_spec)
dateset = tf.data.Dataset.zip((ans(),Sen(),SenWEnt()))
print(dateset.element_spec)