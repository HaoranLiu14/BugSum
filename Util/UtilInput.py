import os
import pandas as pd
from Util import writecsv, DataListListProcess
def SentenceDatainput(filename):
    sentencenumberlist = []
    sentencelist = []
    commentauthor = []
    sentenceInvolved = []
    imptSenInvolveList = []
    imptsentence = ""
    readed_data = open(filename, encoding='UTF-8')
    lines = readed_data.read().split('\n')
    linenum = len(lines)
    for i in range(linenum):
        if (lines[i][:10] == "AuthorName"):
            if (len(commentauthor)!=0):
                sentenceInvolved.append(imptSenInvolveList.copy())
                imptSenInvolveList.clear()
            lines[i] = lines[i][10:]
            lines[i].replace('\'', ' ').replace('_', ' ').replace('.', ' ').replace('-', ' ')
            lines[i] = ' '.join(lines[i].split())
            #commentauthor.append(lines[i][11:])
            commentauthor.append(lines[i][1:-1])
            continue
        if (sentence_num_judgement(lines[i])):
            if (len(sentencenumberlist)!=0):
                sentencelist.append(imptsentence)
                imptsentence = ""
            sentencenumberlist.append('\''+lines[i])
            #imptSenInvolveList.append(lines[i])
            imptSenInvolveList.append(len(sentencenumberlist)-1)
        else:
            imptsentence = imptsentence + lines[i]

    sentencelist.append(imptsentence)
    sentenceInvolved.append(imptSenInvolveList)

    return sentencenumberlist, sentencelist, commentauthor, sentenceInvolved

def readFolderFile(FolderName):
    newfileList = []
    for _, _, files in os.walk(FolderName):
        for file in files:
            newfileList.append(FolderName + "\\" + file)
    return newfileList

def sentence_num_judgement(astring):
    if '.' not in astring:
        return False
    for i in astring:
        if (i <'0' or i>'9') and i!='.':
            return False
    return True

def DataIn(datasetName):
    fileNameList = readFolderFile("Sentence\\" + datasetName)
    writer = pd.ExcelWriter("BugSum_Data_"+datasetName+".xls", engine='xlsxwriter')
    data = {}
    counter = 0
    for fileName in fileNameList:
        #print("fileName", fileName)
        fileNumber = fileName[20:-4]
        print(fileNumber)
        sentencenumberlist, sentencelist, commentauthor, sentenceInvolved = SentenceDatainput(fileName)
        #print("sentencenumberlist", sentencenumberlist)
        #print("sentencelist", sentencelist)
        #print("commentauthor", commentauthor)
        #print("sentenceInvolved", sentenceInvolved)

        data["Sentence"] = sentencelist
        data["SenNumber"] = sentencenumberlist
        data["CommentAuthor"] = commentauthor
        data["ComSenNum"] = DataListListProcess(sentenceInvolved)
        writecsv(writer, fileNumber, data)
        counter = counter + 1
    writer.save()
    writer.close()

#DataIn("AD")
#DataIn("SD")
#DataIn("XD")