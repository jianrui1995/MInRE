import  gensim

f = open("data/TRAIN_SENTENCE.TXT",'r',encoding="utf8")
sentence =f.readlines()
words_list = []
for data in sentence:
    words_list.append(data.split(" "))

model = gensim.models.Word2Vec(words_list,size=350,window=10,min_count=3,workers=32,iter=100000,negative=10)
model.save("model/SemEval-2010 Task8_train_350dim.model")

