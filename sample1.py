import pymongo as pm
import datetime as dt
import random
import gensim as gs

def ValidWord1(word):
    if not ('\u4e00' <= word[0] <= '\u9fff' or '\u4e00' <= word[-1] <= '\u9fff') and \
       not ('\u0041' <= word[0] <= '\u005a' or '\u0041' <= word[-1] <= '\u005a') and \
       not ('\u0061' <= word[0] <= '\u007a' or '\u0061' <= word[-1] <= '\u007a'): # 头尾都不是汉字或字母
        return False
    return True

def ValidWord(word):
    if len(word) <= 1:
        return False
    return ValidWord1(word)

#
mc = pm.MongoClient('mongodb://gongcq:gcq@localhost:27017/text')
# mc = pm.MongoClient('mongodb://gongcq:gcq@192.168.7.83:27017/text')
db = mc['text']

days = 30
docs = db.section.find({'time': {'$gte': dt.datetime.now() - dt.timedelta(days=days)}})
lineBreak = '\n'
paraList = []
idDict = {}

# append documents represented by parse list in to "docParseList"
docList = []
docParseList = []
for doc in docs:
    if doc['masterId'] != '': # it is a duplicate document
        continue
    docList.append(doc)
    docParseList.append(doc['parse'])

# word to id
wordDict = gs.corpora.Dictionary(docParseList)
file = open('wordDict.txt', 'w', encoding='utf-8')
for token, id in wordDict.token2id.items():
    wordDict.id2token[id] = token
    file.write(str(id) + ',' + str(token) + '\n')
file.flush()
file.close()
corpus = [wordDict.doc2bow(doc) for doc in docParseList]
tfIdf = gs.models.TfidfModel(corpus)
tiList = []
for c in range(len(corpus)):
    cor = corpus[c]
    ti = tfIdf[cor]
    tiDict = {}
    for i in range(len(ti)):  # 将词编号还原成词
        ti[i] = (wordDict.id2token[ti[i][0]], ti[i][1])
        tiDict[ti[i][0]] = ti[i][1]
    tiSort = sorted(tiDict.items(), key=lambda d: d[1], reverse=True)  # 按各词的tfidf降序排序
    docList[c]['tiSort'] = tiSort
    tiList.append(tiSort)

# filter words which have low tf-idf
quant = 0.3
docFilterParseList = []
for d in range(len(docList)):
    tiSort = tiList[d]
    remainWordSet = set()
    for i in range(int(quant * len(tiSort))):
        word = tiSort[i][0]
        if ValidWord(word):
            remainWordSet.add(word)
    docFilterParse = []
    docParse = docParseList[d]
    for word in docParse:
        if word in remainWordSet:
            docFilterParse.append(word)
    docFilterParseList.append(docFilterParse)

#
maxId = 0
idDocList = []
idWordDict = {}
wordIdDict = {}
for docFilterParse in docFilterParseList:
    idDoc = []
    for word in docFilterParse:
        if word in wordIdDict.keys():
            wordId = wordIdDict[word]
        else:
            wordId = maxId
            wordIdDict[word] = wordId
            idWordDict[wordId] = word
            maxId += 1
        idDoc.append(wordId)
    idDocList.append(idDoc)
file1 = open('wordIdDict.txt', 'w', encoding='utf-8')
for word, id in wordIdDict.items():
    file1.write(str(id) + ',' + word + '\n')
file1.flush()
file1.close()

interval = 3
numSteps = 7
batchSize = 10
samples = []
for p in range(len(idDocList)):
    para = idDocList[p]
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


aaa = 0