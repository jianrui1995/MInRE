
# 设置标记文件的路径
TRAIN_ANSWER_PATH = "../data/TRAIN_ANSWER.TXT"
# 设置带有实体位置的句子的文件路径
TRAIN_SENTENCE_WITH_ENTITY_PATH = "../data/TRAIN_SENTENCE_WITH_ENTITY.TXT"
# 不带实体位置标记的文本路径
TRAIN_SENTENCE_PATH = "../data/TRAIN_SENTENCE.TXT"
# 训练集的载入路径
TRAIN_PATH = [TRAIN_ANSWER_PATH,TRAIN_SENTENCE_PATH,TRAIN_SENTENCE_WITH_ENTITY_PATH]

# word2vec 模型路径
GLOVE_MODEL_PATH = "../model/glove/vectors.txt"
# 设置单词的最长长度
MAX_WORD_NUM = 100
# 关系类别数量
LABEL_NUM = 19
# 建立类别字典
answer2id = {b'Product-Producer(e2e1)': 0,
             b'Entity-Origin(e2e1)': 1,
             b'Content-Container(e1e2)': 2,
             b'Instrument-Agency(e2e1)': 3,
             b'Message-Topic(e1e2)': 4,
             b'Cause-Effect(e2e1)': 5,
             b'Entity-Origin(e1e2)': 6,
             b'Entity-Destination(e1e2)': 7,
             b'Member-Collection(e1e2)': 8,
             b'Other': 9,
             b'Content-Container(e2e1)': 10,
             b'Cause-Effect(e1e2)': 11,
             b'Instrument-Agency(e1e2)': 12,
             b'Entity-Destination(e2e1)': 13,
             b'Member-Collection(e2e1)': 14,
             b'Product-Producer(e1e2)': 15,
             b'Component-Whole(e1e2)': 16,
             b'Message-Topic(e2e1)': 17,
             b'Component-Whole(e2e1)': 18}

#距离向量的json文件路径
DISTANCE_VECTORY_PATH = "../data/DistanceVectory.json"
