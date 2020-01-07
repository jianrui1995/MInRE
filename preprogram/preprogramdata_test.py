import re
class ReadAndDivede():
    '''
    预处理训练文件，读取并且将信息拆分。
    '''
    def __init__(self):
        self.f = open("../data/TEST_FILE_FULL.TXT", "r", encoding="utf8")
        self.f1 = open("../data/TEST_SENTENCE_WITH_ENTITY.TXT", "w", encoding="utf8")
        self.f2 = open("../data/TEST_ANSWER.TXT", 'w', encoding="utf8")
        self.f3 = open("../data/TEST_SENTENCE.TXT", "w", encoding="utf8")

    def ReadAndDivide(self):
        sentence_list = self.f.readlines()
        for i in range(0,len(sentence_list)):
            if i % 4 ==0:
                sen = "".join(list(re.sub(r".*	","",sentence_list[i]))[1:-2])
                sen = re.sub(r"[,.]","",sen)
                print(sen,file=self.f1,end="\n")
                sen = re.sub(r"</*e[12]>","",sen)
                print(sen,file=self.f3,end="\n")
            if i % 4 ==1:
                print(re.sub(r"[,.]","",sentence_list[i]),file=self.f2,end="")



if __name__ == "__main__":
    RAD = ReadAndDivede()
    RAD.ReadAndDivide()