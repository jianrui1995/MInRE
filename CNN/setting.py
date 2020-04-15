
# 设置输出通道数量
FILTERS = 1000

# 设置词向量维度
WORDVEC = 300
# 设置位置向量维度
POSVEC = 50
# 总体向量维度
VEC = POSVEC*2 + WORDVEC
# 卷积核宽
KERNELWEIGHT = 4

# 训练次数
EPOCH = 1000
# 每训练多少个周期存储
SAVE_N_EPOCH = 25
# 模型存储路径
SAVE_PATH = "../model/CnnModels"
# 模型保存的名字
SAVE_NAME = "cnn"


# 模型载入路径
WEIGHT_PATH = "../model/ModelWeights.ckpt"

# 全连接层1的神经元数量
DENSE_1_NUIM = 500
# 全连接层2的神经元数量
DENSE_2_NUIM = 250
# 全连接层3的神经元数量
DENSE_3_NUIM = 19

# 设置标记文件的路径
TRAIN_ANSWER_PATH = "../data/TRAIN_ANSWER.TXT"
# 设置带有实体位置的句子的文件路径
TRAIN_SENTENCE_WITH_ENTITY_PATH = "../data/TRAIN_SENTENCE_WITH_ENTITY.TXT"
# 不带实体位置标记的文本路径
TRAIN_SENTENCE_PATH = "../data/TRAIN_SENTENCE.TXT"
# 训练集的载入路径
TRAIN_PATH = [TRAIN_ANSWER_PATH,TRAIN_SENTENCE_PATH,TRAIN_SENTENCE_WITH_ENTITY_PATH]

# 设置标记文件的路径
TEST_ANSWER_PATH = "../data/TEST_ANSWER.TXT"
# 设置带有实体位置的句子的文件路径
TEST_SENTENCE_WITH_ENTITY_PATH = "../data/TEST_SENTENCE_WITH_ENTITY.TXT"
# 不带实体位置标记的文本路径
TEST_SENTENCE_PATH = "../data/TEST_SENTENCE.TXT"
# 训练集的载入路径
TEST_PATH = [TEST_ANSWER_PATH,TEST_SENTENCE_PATH,TEST_SENTENCE_WITH_ENTITY_PATH]