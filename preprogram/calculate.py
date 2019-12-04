import preprogram.setting as setting
max = 0
with open(setting.SENTENCE_PATH,"r",encoding="utf8") as f:
    for data in f.readlines():
        word_list = data.strip().split(" ")
        if max < len(word_list):
            max = len(word_list)
print( max)