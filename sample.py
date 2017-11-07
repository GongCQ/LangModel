import pymongo as pm
import datetime as dt
import random

def ValidWord(word):
    # if not ('\u4e00' <= word[0] <= '\u9fff' or '\u4e00' <= word[-1] <= '\u9fff') and \
    #    not ('\u0041' <= word[0] <= '\u005a' or '\u0041' <= word[-1] <= '\u005a') and \
    #    not ('\u0061' <= word[0] <= '\u007a' or '\u0061' <= word[-1] <= '\u007a'): # 头尾都不是汉字或字母
    #     return False
    return True


# mc = pm.MongoClient('mongodb://gongcq:gcq@192.168.5.120:27017/text')
mc = pm.MongoClient('mongodb://gongcq:gcq@localhost:27017/text')
db = mc['text']

days = 8
docs = db.section.find({'time': {'$gte': dt.datetime.now() - dt.timedelta(days=days)}})
lineBreak = '\n'
paraList = []
idParaList = []
wordDict = {}
idDict = {}
maxId = 0
for doc in docs:
    if doc['masterId'] != '': # it is a duplicate document
        continue
    para = []
    idPara = []
    for w in range(len(doc['parse'])):
        word = doc['parse'][w]
        if word in wordDict.keys():
            wordId = wordDict[word]
        else:
            wordId = maxId
            wordDict[word] = wordId
            idDict[wordId] = word
            maxId += 1
        if word == lineBreak or w == len(doc['parse']) - 1:
            if len(para) > 0:
                paraList.append(para)
                idParaList.append(idPara)
                para = []
                idPara = []
        else:
            para.append(word)
            idPara.append(wordId)

interval = 3
numSteps = 15
batchSize = 10
samples = []
for p in range(len(idParaList)):
    para = idParaList[p]
    for s in range(0, len(para) - numSteps, interval):
        x = para[s : s + numSteps]
        y = para[s + 1 : s + numSteps + 1]
        samples.append((x, y))

shufSeq = list(range(len(samples)))
random.shuffle(shufSeq)
batchs = []
batch = ([], [])
for s in shufSeq:
    if len(batch[0]) == batchSize:
        batchs.append(batch)
        batch = ([], [])
    batch[0].append(samples[s][0])
    batch[1].append(samples[s][1])

vocabSize = maxId
print('vocabSize=' + str(vocabSize))

aaa = 0