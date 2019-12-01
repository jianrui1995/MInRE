import gensim

model = gensim.models.Word2Vec.load("../model/SemEval-2010 Task8_train_350dim.model")
print(model["111"])