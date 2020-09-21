import os
import pandas as pd
import spacy
import numpy as np
from bert_serving.client import BertClient

def EmptyCheck(Sentence):
    for i in Sentence:
        if (i>='a' and i<='z') or (i>='A' and i<='Z'):
            return 0
    return 1

def OneHot(num, label_size=7):#changed********************************************************************************************************** label_size=7=> label_size=2
    res = np.zeros(label_size)
    res[num] += 1
    return res

def stopwordinput():
    file = open("StatisticalData\\stoplist.txt", encoding='UTF-8')
    stoplist = file.read().split("\n")
    return stoplist

def stopwordremove(stoplist, text):
    outtext = []
    for word in text:
        if word not in stoplist:
            outtext.append(word)
    return outtext

def preprocess_nltk(text, stoplist):
    import nltk
    import string
    from nltk.corpus import stopwords
    lower = text.lower()

    remove = str.maketrans('','',string.punctuation)
    without_punctuation = lower.translate(remove)
    tokens = nltk.word_tokenize(without_punctuation)#tokens is a list

    #首先获取英文的停用词，在去除掉
    #without_stopwords = [w for w in tokens if not w in stopwords.words('english')]
    #print("Token", tokens)
    without_stopwords = stopwordremove(stoplist, tokens)#without_stopwords is a list

    import nltk.stem
    s = nltk.stem.SnowballStemmer('english')  #参数是选择的语言
    cleaned_text = [s.stem(ws) for ws in without_stopwords]

    imptstr = ' '.join(cleaned_text)

    return imptstr

def stopwordinput():
    file = open("StatisticalData\\stoplist.txt", encoding='UTF-8')
    stoplist = file.read().split("\n")
    return stoplist

'''
def SpecialFormTransform(tokens):

    specialForm = ["isn't", "wasn't", "aren't", "weren't", "haven't", "hasn't", "doesn't", "don't", "didn't"]
    removedForm = ["is not", "was not", "are not", "were not", "have not", "has not", "does not", "do not", "did not"]
    tokensLen = len(tokens)
    newTokens = tokens.copy()
    for i in range(tokensLen):
        token = tokens[i]

        if (token in specialForm):
            pindex = specialForm.index(token)
            newTokens[i] = removedForm[pindex]

    return newTokens
'''

def preprocess_spacy(text, stoplist, nlp):

    doc = nlp(text)
    #tokens = doc.text.split()
    tokens = [t.norm_ for t in doc]
    #transedTokens = SpecialFormTransform(tokens)
    transedtext = " ".join(tokens)
    #print("TransFormed: ", tokens)

    doc = nlp(transedtext)
    tokens = [token.orth_ for token in doc if not token.is_punct | token.is_space] #punction remove
    processedSen = " ".join(tokens)
    #print("Pre Token:", processedSen)

    doc = nlp(processedSen)
    tokens = [word.lemma_ for word in doc]

    #print("LemedSen", " ".join(tokens))

    tokens = stopwordremove(stoplist, tokens)

    LemedSen = " ".join(tokens)
    #print("stop remove", LemedSen)
    return LemedSen

def englishcheck(string):
    if (len(string)<2):
        return False
    for i in string:
        if i>='a' and i<='z' or i>='A' and i<='Z':
            return True
    return False

def writecsv(writer, sheet, data):#sheet is in the form of a list, data is in the form of a dictionary
    df = pd.DataFrame({})
    for (k,j) in data.items():
        df = pd.concat([df, pd.DataFrame({k: j})], axis=1)
    df.to_excel(excel_writer=writer, sheet_name=sheet)

def wirteDataList(dataList, sheetNameList, datasetName):
    filePath = "BugSum_Data_"+datasetName+".xls"
    sheetNum = len(sheetNameList)
    writer = pd.ExcelWriter(filePath, engine='xlsxwriter')
    for i in range(sheetNum):
        writecsv(writer, sheetNameList[i], dataList[i])
    #writer.book.use_zip64()
    writer.save()
    writer.close()

def DataListListProcess(imptList):
    concatedList = []
    imptk = []
    for k in imptList:
        # print("k", k)
        imptk.clear()
        for i in k:
            # print("i", i)
            imptk.append(str(i))
        concatedList.append(" ".join(imptk))
        # print(" ".join(imptk))
    return concatedList

def DataListList2float(inputList):
    answerList = []
    #print("inputList", inputList)
    for imptList in inputList:
        imptList = imptList.split(" ")
        answer = []
        for i in imptList:
            answer.append(float(i))
        answerList.append(answer.copy())
    return answerList

def DataListList2int(inputList):
    answerList = []
    for imptList in inputList:
        imptList = str(imptList)
        imptList = imptList.split(" ")
        answer = []
        for i in imptList:
            if (i==''):
                continue
            answer.append(int(i))
        answerList.append(answer.copy())
    return answerList

def EmptyListCheck(imptList):
    listLen = len(imptList)
    newList = imptList.copy()
    newList.reverse()
    #print("newList", newList)

    for i in range(listLen):
        if newList[i]!='':
            newList = newList[i:]
            break
    newList.reverse()
    return newList

def readcsv(filepath, sheetname, io):
    #df = pd.read_excel(filepath, sheetname, keep_default_na=False)
    df = pd.read_excel(io, sheetname, keep_default_na=False)
    headers_name = list(df.head())[1:]
    data = {}
    for hname in headers_name:
        data[hname] = EmptyListCheck(list(df[hname]))
    #print("data", data)
    return data

def SheetName(filepath):
    xl = pd.ExcelFile(filepath)

    return (list(xl.sheet_names))

def StrList2FloatList(strList):
    floatList = []
    for i in strList:
        floatList.append(float(i))
    return floatList

def StrList2IntList(strList):
    floatList = []
    for i in strList:
        floatList.append(int(i))
    return floatList

def readFileList(DatasetName):
    filelist = open("D:\\GAT\\Reconstruct\\DataSet\\{}\\FileNumberList.txt".format(DatasetName), encoding="utf-8").read().split("\n")
    if len(filelist[-1])==0:
        filelist = filelist[:-1]
    intfileList = []
    for i in filelist:
        intfileList.append(int(i))
    return intfileList

def wordResCounter(ignoreList, percent):
    senNum = len(ignoreList)
    counter = 0
    for i in range(senNum):
        if (ignoreList[i]==0):
            counter = counter + 1
    answer = int(counter*percent)
    if answer == 0:
        answer = 1

    return answer

def wordResGoldenCounter(goldenList, percent):
    senNum = len(goldenList)
    counter = 0
    for i in range(senNum):
        if (goldenList[i] == 1):
            counter = counter + 1
    answer = int(counter * percent)
    if answer>senNum:
        answer = senNum
    if answer == 0:
        answer = 1
    return answer

def AccuracyMeasure(selectedList, goldenList, wordRes):
    tCounter = 0
    senLen = len(goldenList)
    selectedSen = len(selectedList)
    for i in range(senLen):
        if goldenList[i] == 1:
            tCounter = tCounter + 1

    if (tCounter ==0):
        return 1,1

    counter = 0
    for i in selectedList:
        if goldenList[i] == 1:
            counter = counter + 1

    recall = counter*1.0/tCounter
    acc = counter*1.0/selectedSen

    return acc, recall

def answerTypeTrans(answer, senList):
    senNum = len(senList)
    transedAnswer = []
    for i in range(senNum):
        transedAnswer.append(0)

    for i in answer:
        transedAnswer[i] = 1

    return transedAnswer

'''
stoplist = stopwordinput()
preprocess_spacy("There isn't an easy to find way for the end user to set an audio player default that is different than the three \"approved\" applications (Sound Juicer, Rhythmbox, and Banshee) in Ubuntu.", stoplist)
preprocess_spacy("Sorry for the delay in replyin but I haven't looked at \"my\" bugs for a while.", stoplist)
'''