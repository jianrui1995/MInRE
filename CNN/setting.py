# 设置输出通道数量
FILTERS = 1000
# 设置词向量维度
WORDVEC = 350
# 设置位置向量维度
POSVEC = 50
# 总体向量维度
VEC = POSVEC*2 + WORDVEC
# 卷积核宽
KERNELWEIGHT = 4
# 训练次数
EPOCH = 50
# M正的值
M_POS = 2.5
# M负的值
M_NEG = 0.5
# 放大因子
R = 2
# 模型载入路径
WEIGHT_PATH = "../model/ModelWeights.ckpt"
