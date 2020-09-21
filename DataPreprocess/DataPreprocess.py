from Util import readcsv, SheetName, stopwordinput, preprocess_spacy, wirteDataList, EmptyCheck, DataListListProcess, DataListList2float
import re
from bert_serving.client import BertClient
from bert_dpcnn import SOPredict, LoadDPCNNRFC
import math
from Util import DataListList2float, DataListList2int, StrList2FloatList, StrList2IntList, DataListListProcess, wirteDataList
import pandas as pd
import spacy


def DataInput(datasetName):
    filepath = "BugSum_Data_"+str(datasetName)+".xls"
    sheetNameList = SheetName(filepath)

    dataList = []

    io = pd.io.excel.ExcelFile(filepath)

    for sigSheet in sheetNameList:
        #print("sigSheet", sigSheet)
        data = readcsv(filepath, sigSheet, io)

        dataList.append(data.copy())

    return sheetNameList, dataList

def SigDataInput(dataList, i):
    SenList = dataList[i]["Sentence"]
    lemSenList = dataList[i]["LemedSen"]
    senNumberList = dataList[i]["SenNumber"]
    authorList = dataList[i]["CommentAuthor"]
    senInvolvecd = DataListList2int(dataList[i]["ComSenNum"])
    senVecList = DataListList2float(dataList[i]["SenVec"])
    senOPscore = StrList2FloatList(dataList[i]["SenOP"])
    tfidfWordList = dataList[i]["TFIDFWord"]
    tfidfScoreList = StrList2FloatList(dataList[i]["TFIDFScore"])
    ignoreList = StrList2IntList(dataList[i]["NewIgnoredList"])
    #goldenList = StrList2IntList(dataList[i]["GoldenSenNumberList"])

    #print("senVecList", senVecList)
    return SenList, lemSenList, senNumberList, authorList, senInvolvecd, senVecList, senOPscore, tfidfWordList, tfidfScoreList, ignoreList

def BugSumSigDataInput(dataList, i):
    SenList = dataList[i]["Sentence"]
    lemSenList = dataList[i]["LemedSen"]
    senNumberList = dataList[i]["SenNumber"]
    authorList = dataList[i]["CommentAuthor"]
    senInvolvecd = DataListList2int(dataList[i]["ComSenNum"])
    senVecList = DataListList2float(dataList[i]["SenVec"])
    senOPscore = StrList2FloatList(dataList[i]["SenOP"])
    tfidfWordList = dataList[i]["TFIDFWord"]
    tfidfScoreList = StrList2FloatList(dataList[i]["TFIDFScore"])
    ignoreList = StrList2IntList(dataList[i]["NewIgnoredList"])
    goldenList = StrList2IntList(dataList[i]["GoldenSenNumberList"])
    evaluationList = DataListList2int(dataList[i]["EvaluationList"])
    evaluationTimeList = StrList2IntList(dataList[i]["EvaluationTimeList"])
    buildInfoMark = DataListList2int(dataList[i]["IgnoredList"])
    fscoreList = StrList2FloatList(dataList[i]["FscoreList"])
    return SenList, lemSenList, senNumberList, authorList, senInvolvecd, senVecList, senOPscore, tfidfWordList, tfidfScoreList, ignoreList, goldenList, evaluationList, evaluationTimeList, buildInfoMark, fscoreList

def BugSumSigDataWithoutTestInput(dataList, i):
    SenList = dataList[i]["Sentence"]
    lemSenList = dataList[i]["LemedSen"]
    senNumberList = dataList[i]["SenNumber"]
    authorList = dataList[i]["CommentAuthor"]
    senInvolvecd = DataListList2int(dataList[i]["ComSenNum"])
    senVecList = DataListList2float(dataList[i]["SenVec"])
    senOPscore = StrList2FloatList(dataList[i]["SenOP"])
    tfidfWordList = dataList[i]["TFIDFWord"]
    tfidfScoreList = StrList2FloatList(dataList[i]["TFIDFScore"])
    ignoreList = StrList2IntList(dataList[i]["NewIgnoredList"])
    evaluationList = DataListList2int(dataList[i]["EvaluationList"])
    evaluationTimeList = StrList2IntList(dataList[i]["EvaluationTimeList"])
    buildInfoMark = DataListList2int(dataList[i]["IgnoredList"])
    fscoreList = StrList2FloatList(dataList[i]["FscoreList"])
    return SenList, lemSenList, senNumberList, authorList, senInvolvecd, senVecList, senOPscore, tfidfWordList, tfidfScoreList, ignoreList, evaluationList, evaluationTimeList, buildInfoMark, fscoreList


def quoted(Sentence):
    if len(Sentence)>0 and (Sentence[0]==">" or Sentence.find("> ")!=-1):
        return True
    return False

def SignCounter(Sentence):
    counter = 0
    puntc = ['?', ',', '.', '!', ' ', '>']
    for i in Sentence:
        if ((i>='a' and i<='z') or (i>='A' and i<='Z') or i in puntc):
            continue
        else:
            counter = counter + 1
    return counter

def TimeStampRemove(Sentence):
    #result = re.findall('[0-9][0-9][0-9][0-9] [0-9]+[:/][0-9]+[:/][0-9]+',Sentence)
    result = re.findall('[0-9]+[:/-][0-9]+[:/-][0-9]+', Sentence)
    #print("Matched Time", result)
    if len(result)>0:
        return 1
    else:
        return 0

def AnnotatedSentenceCut(Sentence):
    result = re.findall('\*\*\*.*\*\*\*', Sentence)
    #print("Matched Annotated", result)
    if len(result) > 0:
        return 1
    else:
        return 0

def DataRemoveBuildingInfo(dataList, threshold = 50):
    sheetNumber = len(dataList)
    for i in range(sheetNumber):
        #print("dataList[i]", dataList[i])
        dataList[i]["ProcessedSentence"], dataList[i]["IgnoredList"] = SenListRemoveBuildingInfo(dataList[i]["Sentence"], threshold)

    return dataList

def SenListRemoveBuildingInfo(sentences, threshold):
    newSen = sentences.copy()
    ignoredList = []
    senNumber = len(sentences)
    for i in range(senNumber):#ignoredList[i]==1 means that this sentence is building info, and should be ignored during the selection
        ignoredList.append(1)
    for i in range(senNumber):
        sigSen = str(sentences[i])
        #print("sigSen", sigSen)
        if ((len(sigSen.split(" "))>threshold or SignCounter(sigSen)>10) and not quoted(sigSen)):
            #print("Question Sentence", sigSen)

            if (TimeStampRemove(sigSen)):
                #print("Time Stamp Cuted Sentence", sigSen)
                continue

            if (AnnotatedSentenceCut(sigSen)):
                #print("Annotated Sentence Cuted Sentence", sigSen)
                continue

            pos = sigSen.find(":")
            if (pos!=-1 and 5<=len(sigSen[:pos].split(" "))<=threshold):
                sigSen=sigSen[:pos]
                #print("PASS After cut Sentence", sigSen)
                newSen[i]=sigSen
                ignoredList[i] = 0
                continue
            #print("Delete Sentence:", len(i.split(" ")), i)
            #print("Cuted Sentence", sigSen)
            continue
        else:
            #print("PASS Sentence", sigSen)
            newSen[i]=sigSen
            ignoredList[i] = 0
    return newSen, ignoredList

def DataLem(dataList, stoplist):
    sheetNumber = len(dataList)
    nlp = spacy.load("en_core_web_sm")
    for i in range(sheetNumber):
        # print("dataList[i]", dataList[i])
        print("SheetNumber{}、{}".format(i, sheetNumber))
        sentences = dataList[i]["ProcessedSentence"].copy()
        ignoredList = dataList[i]["IgnoredList"].copy()

        #print("sentences", sentences)

        imptSenList = []
        senNum = len(sentences)
        for j in range(senNum):
            sigSen = sentences[j]
            newSen = preprocess_spacy(sigSen, stoplist, nlp)
            if (EmptyCheck(newSen)):
                ignoredList[j] = 1
                imptSenList.append("***EMPTY_SENTENCE***")
            else:
                imptSenList.append(newSen)

        dataList[i]["LemedSen"] = imptSenList.copy()
        dataList[i]["IgnoredList"] = ignoredList.copy()
    return dataList

def Bert2Vec(dataList):
    bc = BertClient(check_length=False)
    sheetLen = len(dataList)
    imptSenVecList = []
    for i in range(sheetLen):
        print("Finish:", i/sheetLen, "Percent")
        targetList = list(dataList[i]["LemedSen"])
        sentenceVecList = bc.encode(targetList)
        #print(sentenceVecList.type())
        sentenceVecList = sentenceVecList.tolist()
        #print("sentenceVecList", len(sentenceVecList[0]))
        dataList[i]["SenVec"] = DataListListProcess(sentenceVecList)
        print("sentenceVecList", len(sentenceVecList), len(sentenceVecList[0]))
    return dataList

def DataOPscore(dataList):
    sheetNum = len(dataList)
    dpcnn, rfc = LoadDPCNNRFC()
    for i in range(sheetNum):
        print("OPscore Processing:", i, "/", sheetNum)
        senVecList = DataListList2float(dataList[i]["SenVec"])
        #print("senVecList", len(senVecList), len(senVecList[1]))
        OPscoreList = SOPredict(senVecList, dpcnn, rfc)
        dataList[i]["SenOP"] = OPscoreList.tolist()
    return dataList

def TFIDFCounter(dataList):
    sheetNum = len(dataList)

    tfWordDic = {} #in single file
    idfWordDic = {} #in entire database

    tfDicList = []

    for i in range(sheetNum):
        senList = dataList[i]["LemedSen"]
        tfToalNum = 0
        for sigSen in senList:
            wordList = sigSen.split(" ")
            for sigWord in wordList:
                tfToalNum = tfToalNum + 1
                if sigWord not in tfWordDic:
                    tfWordDic[sigWord] = 1
                    if sigWord not in idfWordDic:
                        idfWordDic[sigWord] = 1
                    else:
                        idfWordDic[sigWord] = idfWordDic[sigWord] + 1
                else:
                    tfWordDic[sigWord] = tfWordDic[sigWord] + 1
        for k, v in tfWordDic.items():
            tfWordDic[k] = float(v)/float(tfToalNum)
        #print("tfWordDic", tfWordDic)
        tfDicList.append(tfWordDic.copy())
        tfWordDic.clear()

    for k, v in idfWordDic.items():
        idfWordDic[k] = math.log(sheetNum/(v+1))
    #print("idfWordDic", idfWordDic)

    for i in range(sheetNum):
        tfidfWordList = []
        tfidfScoreList = []
        for k, v in tfDicList[i].items():
            tfidfWordList.append(k)
            tfidfScoreList.append(v*idfWordDic[k])
        dataList[i]["TFIDFWord"] = tfidfWordList.copy()
        dataList[i]["TFIDFScore"] = tfidfScoreList.copy()
        tfidfWordList.clear()
        tfidfScoreList.clear()

    return dataList

def GoldenNumberMatch(goldenSenNumberList, senNumberList):
    #print("senNumberList", senNumberList)
    #print("goldenSenNumberList", goldenSenNumberList)
    xlsGoldenList = []
    answer = []

    goldenNum = len(goldenSenNumberList)
    for i in range(goldenNum):
        xlsGoldenList.append('\''+goldenSenNumberList[i])

    senNum = len(senNumberList)
    for i in range(senNum):
        answer.append(0)

    for i in xlsGoldenList:
        #print("answer", answer)
        #print("senNumberList", senNumberList.index(i))
        answer[senNumberList.index(i)] = 1
    #print("answer", answer)
    return answer

def GoldenSet(dataList, datasetName, sheetNameList):
    filepath = datasetName + "GoldenSet.txt"
    f = open(filepath)
    goldenList = f.readlines()
    startSheetName = goldenList[0][:goldenList[0].find(',')]
    goldenSenNumberList = []
    GoldenList = []
    for sigLine in goldenList:
        sheetName = sigLine[:sigLine.find(',')]
        goldenSenNumber = sigLine[sigLine.find(',')+1:-1]
        goldenSenNumber = goldenSenNumber.strip()

        if sheetName!=startSheetName:
            dataIndex = sheetNameList.index(startSheetName)
            print("Processing:", startSheetName)
            GoldenList = GoldenNumberMatch(goldenSenNumberList, dataList[dataIndex]["SenNumber"])
            #print("GoldenList", GoldenList)
            dataList[dataIndex]["GoldenSenNumberList"]=GoldenList.copy()
            #print("goldenSenNumberList", goldenSenNumberList)
            goldenSenNumberList.clear()
            startSheetName = sheetName

        goldenSenNumberList.append(goldenSenNumber)
    #print("Here")
    dataIndex = sheetNameList.index(startSheetName)
    GoldenList = GoldenNumberMatch(goldenSenNumberList, dataList[dataIndex]["SenNumber"])
    dataList[sheetNameList.index(sheetName)]["GoldenSenNumberList"] = GoldenList.copy()
    #print("Number", sheetNameList.index(sheetName))
    #print("goldenSenNumberList", goldenSenNumberList)
    goldenSenNumberList.clear()
    return dataList

def DataList2Str(dataList):
    sheetLen = len(dataList)
    for i in range(sheetLen):
        imptList = dataList[i]["LemedSen"]
        newList = []
        for sigSen in imptList:
            newList.append(str(sigSen))
        dataList[i]["LemedSen"] = newList.copy()
    return dataList


def DataPreprocess(datasetName):
    sheetNameList, dataList = DataInput(datasetName)


    print("Start")
    dataList = DataRemoveBuildingInfo(dataList, threshold=50)
    print("Finish Remove")
    stoplist = stopwordinput()
    dataList = DataLem(dataList, stoplist)
    wirteDataList(dataList, sheetNameList, datasetName)
    print("Finish Lem")
    
    dataList = DataList2Str(dataList)

    dataList = Bert2Vec(dataList)
    wirteDataList(dataList, sheetNameList, datasetName)
    print("Finish Bert")

    dataList = DataOPscore(dataList)
    wirteDataList(dataList, sheetNameList, datasetName)
    print("Finish OPscore")

    dataList = TFIDFCounter(dataList)
    wirteDataList(dataList, sheetNameList, datasetName)
    print("Finish TFIDF")
    
    #dataList = GoldenSet(dataList, datasetName, sheetNameList)
    #print("Finish GoldenSet")

    wirteDataList(dataList, sheetNameList, datasetName)

