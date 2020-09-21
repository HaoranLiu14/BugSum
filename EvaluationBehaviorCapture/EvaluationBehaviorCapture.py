import math
from Util import DataListList2float, DataListList2int, StrList2FloatList, StrList2IntList, DataListListProcess, wirteDataList
from DataPreprocess import DataInput, SigDataInput

def EvaluationBehaviorCap(dataList, sheetName, domainwordThreshold):
    sheetNum = len(dataList)

    for i in range(sheetNum):
        print("Processing :", sheetName[i])
        SenList, lemSenList, senNumberList, authorList, senInvolvecd, senVecList, senOPscore, tfidfWordList, tfidfScoreList, ignoreList = SigDataInput(dataList, i)

        domainwordlist = DomainWordSelection(tfidfWordList, tfidfScoreList, domainwordThreshold)

        evaluationList, evaluationTimeList, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, NewEvaluationUnion = directlycite(SenList, senInvolvecd)
        DirectQuoteList = NewEvaluationUnion.copy()

        EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList, ignoreList = CommentNumberCite(SenList, evaluationTimeList, senInvolvecd, domainwordlist, evaluationList, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, ignoreList)
        #print("evaluationList", evaluationList)
        evaluationList, evaluationTimeList, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, domainwordlist = DescriptionCite(senInvolvecd, SenList, domainwordlist, evaluationTimeList, evaluationList, EvaluatorSentenceNumberList, CombinedQuotedSentenceList)
        #print("evaluationList", evaluationList)
        evaluationList, evaluationTimeList, evaluatorSentenceNumberList, newCombinedQuotedSentenceList, ignoreList = AuthorCite(SenList, authorList, senInvolvecd, evaluationList, evaluationTimeList, domainwordlist, ignoreList)  # evaluatorSentenceNumberList is also needed.
        #print("evaluationList", evaluationList)
        evaluationTimeList, evaluationList, newCombinedQuotedSentenceList, EvaluatorSentenceNumberList, AmbigousRstatisticNum = HighAscoreSenEval_KeyWordBased(senInvolvecd, SenList, evaluationTimeList, evaluationList, evaluatorSentenceNumberList, newCombinedQuotedSentenceList, senOPscore, domainwordlist)
        #print("evaluationList", evaluationList)
        evaluatedList = EvaluatedList(evaluationList)
        FscoreList = Fscore(evaluatedList, senOPscore)
        senMark = sentenceMarkList(DirectQuoteList, SenList, ignoreList)

        dataList[i]["EvaluationList"] = DataListListProcess(evaluationList)
        dataList[i]["EvaluationTimeList"] = evaluationTimeList
        dataList[i]["NewIgnoredList"] = senMark
        dataList[i]["FscoreList"] = FscoreList
    return dataList

def directlycite(sentencelist, sentenceInvolved):
    commentnum = len(sentenceInvolved)
    #print("sentencelistLen", len(sentencelist))
    #print("sentencelist", sentencelist)
    evaluationBehaviorList = []
    QuotedSentenceList = []
    for i in range(commentnum):
        for sennumber in sentenceInvolved[i]:
            #print("Sen", sentencelist[int(sennumber)])
            if ('>' in sentencelist[int(sennumber)]):
                OneSentenceQuotedSentenceList = quoted_sentence(sentencelist[sennumber])
                #print("OneSentenceQuotedSentenceList", OneSentenceQuotedSentenceList)
                OneSentenceQuotedSentenceList = quotedSentenceListCheck(OneSentenceQuotedSentenceList)#remove empty list
                if (len(OneSentenceQuotedSentenceList)==0):
                    continue
                #print("OneSentenceQuotedSentenceList", OneSentenceQuotedSentenceList)
                #print("sennumber", sennumber)
                QuotedSentenceList.append(OneSentenceQuotedSentenceList.copy())
                evaluationBehaviorList.append(sennumber)#contain the sentence number of quoter sentences, such as the number of sentence: ">a, >b, >c": 15

    #print("evaluationBehaviorList", evaluationBehaviorList)
   # print("QuotedSentenceList", QuotedSentenceList)
    EvaluationListUnion, CombinedQuotedSentenceList, EvaluatorSentenceNumberList = findEvaluatorSentence(evaluationBehaviorList, QuotedSentenceList, sentencelist, sentenceInvolved)
    #print("EvaluationListUnion", EvaluationListUnion)
    #print("CombinedQuotedSentenceList", CombinedQuotedSentenceList)
    #print("EvaluatorSentenceNumberList", EvaluatorSentenceNumberList)

    newEvaluatorSentenceNumberList = []
    newCombinedQuotedSentenceList = []

    evaluationList, evaluationTimeList, NewEvaluationUnion, newEvaluatorSentenceNumberList, newCombinedQuotedSentenceList = findquotedsentence(EvaluationListUnion, CombinedQuotedSentenceList, EvaluatorSentenceNumberList, sentencelist, newEvaluatorSentenceNumberList, newCombinedQuotedSentenceList)
    #return evaluationList, evaluationTimeList, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, NewEvaluationUnion, newEvaluatorSentenceNumberList, newCombinedQuotedSentenceList
    return evaluationList, evaluationTimeList, newEvaluatorSentenceNumberList, newCombinedQuotedSentenceList, NewEvaluationUnion

def DescriptionCite(sentenceInvolved, sentencelist, domainwordlist, evaluationTimeList, evaluationList, EvaluatorSentenceNumberList, CombinedQuotedSentenceList):

    commentnum = len(sentenceInvolved)
    #print("domainwordlist", domainwordlist)

    for DescSentenceNumber in sentenceInvolved[0]:
        imptDomainWordList = []
        for Dword in domainwordlist:
            Dword = str(Dword)
            #print("Dword", Dword)
            #print("sentencelist[DescSentenceNumber]", sentencelist[DescSentenceNumber])
            if Dword in sentencelist[DescSentenceNumber]:
                imptDomainWordList.append(Dword)
        for commentnumber in range(1,commentnum):
            senNumInComment = len(sentenceInvolved[commentnumber])
            for i in range(senNumInComment):
                sentencenumber = sentenceInvolved[commentnumber][i]
                for TargetDomainWord in imptDomainWordList:
                    if (TargetDomainWord in sentencelist[sentencenumber]):#This "sentencenumber" is also have certain value as the evaluator sentence
                        EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList = addAEvaluatinBehavior(DescSentenceNumber, sentencenumber, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList)

    return evaluationList, evaluationTimeList, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, domainwordlist

def HighAscoreSenEval_KeyWordBased(sentenceInvolved, sentencelist, evaluationTimeList, evaluationList, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, Ascore, domainwordlist):
    consize = 1
    Arange = 0.8
    senNum = len(sentencelist)
    comNum = len(sentenceInvolved)
    skipSentenceList, skipCommentList = senCheckForKeyWordAscreSenEval(sentencelist, sentenceInvolved, EvaluatorSentenceNumberList)

    #print("Ascore", Ascore)
    #print("skipSentenceList", skipSentenceList)
    #print("skipCommentList", skipCommentList)

    AmbigousRstatistic = []
    for i in range(comNum):
        AmbigousRstatistic.append(0)

    for comNumber in range(1, comNum):
        if (skipCommentList[comNumber]==1):
            continue
        for senNumber in sentenceInvolved[comNumber]:
            if (skipSentenceList[senNumber]==1):
                continue
            #if Ascore[senNumber] < (-0.4+Arange) or Ascore[senNumber] > (1.6-Arange):
            if Ascore[senNumber] < (-1 + Arange) or Ascore[senNumber] > (1 - Arange):
                flag = 0
                #print("Find One!", sentencelist[senNumber])
                for j in range(max(0,comNumber-1), max(0,comNumber-consize-1), -1):
                    counter = 0
                    #print(sentenceInvolved[j])
                    for candiSenNumber in sentenceInvolved[j]:
                        counter = counter+1
                        if (counter>=4):
                            break
                        KeyWord = SenDomainWordCheck(domainwordlist, sentencelist[senNumber], sentencelist[candiSenNumber])

                        if (KeyWord!="False!!!!"):
                            #print("KeyWord here bitch", KeyWord)
                            if (candiSenNumber not in evaluationList[senNumber]): #This have to be tune back
                                #statistic code1
                                AmbigousRstatistic[comNumber] = 1
                                AmbigousRstatistic[j] =1
                                # statistic code2
                                #recordDiffThreshold(KeyWord, sentencelist, senNumber, candiSenNumber, sentenceInvolved, fileName, Ascore)
                                # statistic finish
                                #print(Ascore[senNumber], "Capture 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", sentencelist[senNumber], "evaluating!!!!!!!!!!!!!!!!!",sentencelist[candiSenNumber])
                                EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList = addAEvaluatinBehavior(candiSenNumber, senNumber, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList)
                                flag = 1
                    if flag==1:
                        break

    AmbigousRstatisticNum = 0
    for i in AmbigousRstatistic:
        if i==1:
            AmbigousRstatisticNum = AmbigousRstatisticNum + 1

    return evaluationTimeList, evaluationList, CombinedQuotedSentenceList, EvaluatorSentenceNumberList, AmbigousRstatisticNum

def sentenceMarkList(CombinedQuotedSentenceList, sentenceList, ignoreList):
    senMark = ignoreList.copy()
    '''
    senNum = len(sentenceList)
    for i in range(senNum):
        senMark.append(0)
    '''

    #print(CombinedQuotedSentenceList, CombinedQuotedSentenceList)
    for i in CombinedQuotedSentenceList:
        senMark[i] = 1

    return senMark

def SenDomainWordCheck(domainwordList, sena, senb):
    #print(domainwordList)
    for i in domainwordList:
        if i in sena and i in senb:
            #print("Domainword:", i)
            return i
    return "False!!!!"

def senCheckForKeyWordAscreSenEval(sentencelist, sentenceInvolved, EvaluatorSentenceNumberList):
    ComNum = len(sentenceInvolved)
    senNum = len(sentencelist)
    skipSentenceList = []
    skipCommentList = []
    for s in range(senNum):
        skipSentenceList.append(0)
    for s in range(ComNum):
        skipCommentList.append(0)

    #print("Here skipSentenceList", len(skipSentenceList))
    #"Here skipCommentList", len(skipCommentList))

    for comNumber in range(ComNum):
        for senNumber in sentenceInvolved[comNumber]:
            if senNumber in EvaluatorSentenceNumberList:
                #skipSentenceList[senNumber] = 1

                skipCommentList[comNumber] = 1
                break

            if ">" in sentencelist[senNumber] or "?" in sentencelist[senNumber]:
                skipSentenceList[senNumber] = 1
    return skipSentenceList, skipCommentList

def quoted_sentence(sentence):#return quoted sentences, for example, if >a, >b, >c. then it return ["a", "b", "c"]
    sentenceLen = len(sentence)
    quotedSentenceList = []
    citestart = sentence.find(">")
    if citestart == sentenceLen:
        return quotedSentenceList
    for i in range(citestart,sentenceLen-1):
        if sentence[i+1]!=' ':
            citestart = i
            break
    citeend = sentence.find(">",citestart+1)
    while(citeend!=-1):
        diff = citeend - citestart
        if (diff <= 2):
            citestart = citeend
            citeend = sentence.find(">", citestart + 1)
        else:
            if (diff>18):
                quotedSentenceList.append(sentence[citestart+2 : citestart + 20])
            else:
                quotedSentenceList.append(sentence[citestart+2 : citeend])
            citestart = citeend
            citeend = sentence.find(">", citestart + 1)
    if (sentenceLen > citestart + 18):
        quotedSentenceList.append(sentence[citestart+2 : citestart + 20])
    else:
        quotedSentenceList.append(sentence[citestart+2:])
    return quotedSentenceList

def quotedSentenceListCheck(quotedSentenceList):#remove the empty list, such as: ['']
    listnum = len(quotedSentenceList)
    for sentencenum in range(listnum):
        if (len(quotedSentenceList[sentencenum])<=0):
            del quotedSentenceList[sentencenum]
    return quotedSentenceList

def ListUnion(a, b):
    for i in b:
        a.append(i)
    return a

def DomainWordSelection(tfidfWordList, tfidfScoreList, domainWordThreshold):
    wordNum = len(tfidfWordList)
    domainWordList = []
    for i in range(wordNum):
        if (tfidfScoreList[i]>=domainWordThreshold and len(str(tfidfWordList[i]))>1):
            domainWordList.append(tfidfWordList[i])
    #print("Selected DomainwordList")
    return domainWordList

def AuthorCite(sentencelist, commentauthor, sentenceInvolved, evaluationList, evaluationTimeList, domainwordlist, ignoreList):
    authorList = []
    for i in commentauthor:
        if i not in authorList:
            authorList.append(i)

    senNum = len(sentencelist)
    comNum = len(commentauthor)

    evaluatorSentenceNumberList = []
    #candidateListPAD = []
    CombinedQuotedSentenceList = []

    for authorFullName in authorList:
        authorNameList = authorFullName.split()
        #print("authorFullName", authorFullName)

        for authorName in authorNameList:
            #print("authorName", authorName)
            if (len(authorName)<3):
                continue
            authorName = authorName.lower()
            for j in range(senNum):
                sentence = sentencelist[j]
                #print("sentence", sentence.lower())
                if (authorName in sentence.lower()):
                    #print("yes!!!!!!!!!!!!!!!!!!")
                    ComNumber = FindComNumberWithSenNumber(j, sentenceInvolved)
                    #print(j, sentenceInvolved)
                    #print("ComNumber", ComNumber)
                    ComRange = range(ComNumber)
                    for z in reversed(ComRange):
                        if (authorName in commentauthor[z].lower()):
                            if (checkEvBetweenComments(ComNumber, z, evaluationList, sentenceInvolved)):
                                continue
                            elif (CheckComForEvaluatedSentence(z, evaluationTimeList, sentenceInvolved)):
                                for EvaluatedSenNum in sentenceInvolved[z]:
                                    if evaluationTimeList[EvaluatedSenNum] > 0:
                                        if(EvaluatedSenNum>j):
                                            continue
                                        evaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList = addAEvaluatinBehavior(
                                            EvaluatedSenNum, j, evaluatorSentenceNumberList,
                                            CombinedQuotedSentenceList, evaluationList, evaluationTimeList)
                                        #ignoreList[j] = 1
                            else:
                                senInCommentNum = len(sentenceInvolved[z])
                                if (senInCommentNum > 0):
                                    for j in range(senInCommentNum):
                                        if (CheckShareDomainWordBetweenSen(sentenceInvolved[z][j], ComNumber, sentencelist, domainwordlist)):
                                        #if (CheckShareDomainWordBetweenSenandCom(sentenceInvolved[z][j], ComNumber, sentencelist, domainwordlist,sentenceInvolved)):
                                            if (sentenceInvolved[z][j] > j):
                                                continue
                                            evaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList = addAEvaluatinBehavior(
                                                sentenceInvolved[z][j], j,
                                                evaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList,
                                                evaluationTimeList)
                                            #ignoreList[j] = 1
                            #addAEvaluatinBehavior()
                            '''
                            if (j not in evaluatorSentenceNumberList):
                                if ("wrote" in sentencelist[j] or "schrieb" in sentencelist[j]):
                                    continue
                                #print("match!!!!!!!!!!!!!!!!!!!")
                                evaluatorSentenceNumberList.append(j)
                                candidateListPAD.append(z)
                            for k in sentenceInvolved[z]:
                                if (evaluationTimeList[k] != 0):
                                    if (k not in evaluationList[j]):
                                        evaluationList[j].append(k)
                                        evaluationTimeList[k] = evaluationTimeList[k] + 1
                            '''

    return evaluationList, evaluationTimeList, evaluatorSentenceNumberList, CombinedQuotedSentenceList, ignoreList



def DirectQuoteEvaluatorLocate(OneCombinedQuotedSentenceList, sentenceInvolved):
    candidateEvaluatorComNumAfter = FindComNumberWithSenNumber(OneCombinedQuotedSentenceList[-1]+1, sentenceInvolved)
    candidateEvaluatorComNumBefore = FindComNumberWithSenNumber(OneCombinedQuotedSentenceList[0]-1, sentenceInvolved)
    quoterComNumber = FindComNumberWithSenNumber(OneCombinedQuotedSentenceList[0], sentenceInvolved)
    if (candidateEvaluatorComNumAfter == quoterComNumber):
        return OneCombinedQuotedSentenceList[-1]+1
    elif (candidateEvaluatorComNumBefore == quoterComNumber):
        return OneCombinedQuotedSentenceList[0]-1
    else:
        return -1


def findEvaluatorSentence(evaluationBehaviorList, QuotedSentenceList, senList, sentenceInvolved): # This function combines neighboring evaluation behaviors, and find the evaluator sentence that used to assess the believability.
    senNum=len(senList)
    evaluationBehaviorNumber = len(evaluationBehaviorList)

    oneEvaluationBehavior = []
    EvaluationListUnion = []
    EvaluatorSentenceNumberList = []
    CombinedQuotedSentenceList = []
    OneCombinedQuotedSentenceList = []

    if (evaluationBehaviorNumber==0):
        return EvaluationListUnion, CombinedQuotedSentenceList, EvaluatorSentenceNumberList

    #oneEvaluationBehavior.append(QuotedSentenceList[0])
    oneEvaluationBehavior = ListUnion(oneEvaluationBehavior, QuotedSentenceList[0].copy())

    for itime in range(len(QuotedSentenceList[0])):
        #print("add", evaluationBehaviorList[0])
        OneCombinedQuotedSentenceList.append(evaluationBehaviorList[0])

    for i in range(1, evaluationBehaviorNumber):
        if (evaluationBehaviorList[i] == evaluationBehaviorList[i-1] + 1 and FindComNumberWithSenNumber(evaluationBehaviorList[i], sentenceInvolved)==FindComNumberWithSenNumber(evaluationBehaviorList[i-1] + 1, sentenceInvolved)):
            oneEvaluationBehavior = ListUnion(oneEvaluationBehavior, QuotedSentenceList[i].copy())
            for itime in range(len(QuotedSentenceList[i])):
                #print("add", evaluationBehaviorList[i])
                OneCombinedQuotedSentenceList.append(evaluationBehaviorList[i])
        else:
            evaluatorSenNum = DirectQuoteEvaluatorLocate(OneCombinedQuotedSentenceList, sentenceInvolved)
            if (evaluatorSenNum != -1):
                EvaluatorSentenceNumberList.append(evaluatorSenNum)

                EvaluationListUnion.append(oneEvaluationBehavior.copy())#quoted sentence number list
                CombinedQuotedSentenceList.append(OneCombinedQuotedSentenceList.copy())

            oneEvaluationBehavior.clear()
            OneCombinedQuotedSentenceList.clear()
            oneEvaluationBehavior = ListUnion(oneEvaluationBehavior, QuotedSentenceList[i].copy())
            for itime in range(len(QuotedSentenceList[i])):
                #print("add", evaluationBehaviorList[i])
                OneCombinedQuotedSentenceList.append(evaluationBehaviorList[i])

    evaluatorSenNum = DirectQuoteEvaluatorLocate(OneCombinedQuotedSentenceList, sentenceInvolved)
    if (evaluatorSenNum != -1):
        EvaluatorSentenceNumberList.append(evaluatorSenNum)

        EvaluationListUnion.append(oneEvaluationBehavior.copy())  # quoted sentence number list
        CombinedQuotedSentenceList.append(OneCombinedQuotedSentenceList.copy())

    '''
    if (evaluationBehaviorList[evaluationBehaviorNumber - 1] + 1 < senNum):
        EvaluatorSentenceNumberList.append(evaluationBehaviorList[evaluationBehaviorNumber - 1] + 1)
    else:
        if (len(CombinedQuotedSentenceList)>0):
            CombinedQuotedSentenceList.pop()
    '''
    #EvaluationListUnion [['there are ', 'rather con', 'pascal, yo', 'so are you', 12, 'in the las']]
    #CombinedQuotedSentenceList [[12, 13, 14, 15, 16]]
    #EvaluatorSentenceNumberList [17]
    #print(CombinedQuotedSentenceList, EvaluatorSentenceNumberList)
    return EvaluationListUnion, CombinedQuotedSentenceList, EvaluatorSentenceNumberList

def FindIndexWithComNumber(CommentNumber, sentencenumber, sentenceInvolved):
    pos = sentenceInvolved[CommentNumber].index(sentencenumber)
    return pos

def FindComNumberWithSenNumber(Sennumber, sentenceInvolved):
    comnum = len(sentenceInvolved)
    for i in range(comnum):
        if (Sennumber in sentenceInvolved[i]):
            return i

def findquotedsentence(EvaluationListUnion, CombinedQuotedSentenceList, EvaluatorSentenceNumberList, sentencelist, newEvaluatorSentenceNumberList, newCombinedQuotedSentenceList):
    #print("EvaluationListUnion and CombinedQuotedSentenceList", len(EvaluationListUnion), len(CombinedQuotedSentenceList))
    sentencenum = len(sentencelist)

    #print("EvaluationListUnion", EvaluationListUnion)
    #print("CombinedQuotedSentenceList", CombinedQuotedSentenceList)
    #print("EvaluatorSentenceNumberList", EvaluatorSentenceNumberList)

    evaluationList = []# storing the evaluation behavior linked by directly quote. e.g. 8 evaluating 1,2,3, then evaluationList[8] = {1,2,3}
    evaluationTimeList = []#storing each sentence cited time (e.g. 2 is evaluated by 8,9,10, then evaluationTimeList[i] == 3)

    emptylist = []
    for i in range(sentencenum):
        evaluationList.append(emptylist.copy())
        evaluationTimeList.append(0)

    UnionNum = len(CombinedQuotedSentenceList)
    #print("CombinedQuotedSentenceList", CombinedQuotedSentenceList)

    NewEvaluationUnion = []

    for i in range(UnionNum):
        imptSenNum = len(CombinedQuotedSentenceList[i])
        flag = 0
        for j in range(imptSenNum):
            quotedSenNumber = CombinedQuotedSentenceList[i][j]
            quotedSen = EvaluationListUnion[i][j]
            #print("quotedSenNumber", quotedSenNumber)
            #quotedSen = sentencelist[quotedSenNumber]
            for z in range(quotedSenNumber):
                if (quotedSen in sentencelist[z]):
                    flag = 1
                    #print(EvaluatorSentenceNumberList[i], z)
                    newEvaluatorSentenceNumberList, newCombinedQuotedSentenceList, evaluationList, evaluationTimeList = addAEvaluatinBehavior(z, EvaluatorSentenceNumberList[i], newEvaluatorSentenceNumberList, newCombinedQuotedSentenceList, evaluationList, evaluationTimeList)
                    '''
                    if (z not in evaluationList[EvaluatorSentenceNumberList[i]]):
                        evaluationList[EvaluatorSentenceNumberList[i]].append(z)
                        evaluationTimeList[z] = evaluationTimeList[z] + 1
                    '''
                    break
        if (flag == 1):
            for k in CombinedQuotedSentenceList[i]:
                NewEvaluationUnion.append(k)

    return evaluationList, evaluationTimeList, NewEvaluationUnion, newEvaluatorSentenceNumberList, newCombinedQuotedSentenceList

def DirectlyQuotedCommentNumberDetect(evaluationList, sentenceInvolved):
    CommentNumber = len(sentenceInvolved)

    CommentQuotedList = []  # Contain the quoted sentence index of each comment.(e.g. if comment1 contain [1,2,3,4] and 4 is quoted, then CommentQuotedList[1] = [4])
    imptempty = []
    for i in range(CommentNumber):
        CommentQuotedList.append(imptempty.copy())

    for SingleList in evaluationList:
        if (len(SingleList)>0):
            for snumber in SingleList:
                for i in range(CommentNumber): #comment number
                    if (snumber in sentenceInvolved[i]):
                        CommentQuotedList[i].append(snumber)
                        break
    return CommentQuotedList

def checkNumber(numa):
    for i in numa:
        if i < '0' or i > '9':
            return False
    else:
        return True

def CheckCommentNumber(stringa):
    CNcitestr = "comment #"
    spacestr = " "
    ComNumber = -1
    pos = stringa.find(CNcitestr)
    if (pos == -1):
        return ComNumber
    else:
        spaceend = stringa.find(spacestr, pos+10)
        if (spaceend == -1):
            if (checkNumber(stringa[pos+9:])):
                ComNumber = int(stringa[pos+9:])
            else:
                ComNumber = -1
        else:
            if (checkNumber(stringa[pos + 9:spaceend])):
                ComNumber = int(stringa[pos + 9:spaceend])
            else:
                ComNumber = -1
        return ComNumber

def CommentNumberCite(sentencelist, evaluationTimeList, sentenceInvolved, domainwordlist, evaluationList, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, ignoreList):

    commentnum = len(sentenceInvolved)
    for i in range(commentnum):
        for sennumber in sentenceInvolved[i]:
            ContainedCNCite = CheckCommentNumber(sentencelist[int(sennumber)])
            if (ContainedCNCite == -1):
                continue
            else:
                #this evaluate based on key word sharing
                if (checkEvBetweenComments(i, ContainedCNCite, evaluationList, sentenceInvolved)):
                    continue
                elif (CheckComForEvaluatedSentence(ContainedCNCite, evaluationTimeList, sentenceInvolved)):
                    for EvaluatedSenNum in sentenceInvolved[ContainedCNCite]:
                        if evaluationTimeList[EvaluatedSenNum]>0:
                            EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList = addAEvaluatinBehavior(EvaluatedSenNum, int(sennumber), EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList)
                            #ignoreList[int(sennumber)] = 1
                else:
                    senInCommentNum = len(sentenceInvolved[ContainedCNCite])
                    if (senInCommentNum > 0):
                        for j in range(senInCommentNum):
                            if (CheckShareDomainWordBetweenSen(sentenceInvolved[ContainedCNCite][j], sennumber, sentencelist, domainwordlist)):
                            #if (CheckShareDomainWordBetweenSenandCom(sentenceInvolved[ContainedCNCite][j], i, sentencelist, domainwordlist, sentenceInvolved)):
                                EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList = addAEvaluatinBehavior(sentenceInvolved[ContainedCNCite][j], sennumber, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList)
    return EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList, ignoreList

def addAEvaluatinBehavior(i, j, EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList): #i is evaluated by j
    evaluationTimeList[i] = evaluationTimeList[i] + 1
    if (j not in EvaluatorSentenceNumberList):
        EvaluatorSentenceNumberList.append(j)
        imptlist = []
        imptlist.append(i)
        CombinedQuotedSentenceList.append(imptlist.copy())
    else:
        pos = EvaluatorSentenceNumberList.index(j)
        #print("pos", pos)
        #print("EvaluatorSentenceNumberList", EvaluatorSentenceNumberList)
        #print("j", j)
        #print(len(CombinedQuotedSentenceList),"CombinedQuotedSentenceList", CombinedQuotedSentenceList)
        CombinedQuotedSentenceList[pos].append(i)
    if (i not in evaluationList[j]):
        evaluationList[j].append(i)
    return EvaluatorSentenceNumberList, CombinedQuotedSentenceList, evaluationList, evaluationTimeList

def checkEvBetweenComments(a, b, evaluationList, sentenceInvolved):#check if there is a already evaluate b
    bsennumlist = sentenceInvolved[b]
    for i in sentenceInvolved[a]:
        for j in evaluationList[i]:
            if j in bsennumlist:
                return True
    return False

def CheckComForEvaluatedSentence(comnumber, evaluationTimeList, sentenceInvolved):
    for i in sentenceInvolved[comnumber]:
        if evaluationTimeList[i]>0:
            return True
    return False

def CheckShareDomainWordBetweenSen(sena, senb, sentencelist, domainwordlist):
    senalist = sentencelist[sena].split()
    senblist = sentencelist[senb].split()
    for i in domainwordlist:
        if i in senalist and i in senblist:
            return True
    return False

def CheckShareDomainWordBetweenSenandCom(sena, comb, sentencelist, domainwordlist, sentenceInvolved):
    senalist = sentencelist[sena].split()
    for i in sentenceInvolved[comb]:
        senblist = sentencelist[i].split()
        for j in domainwordlist:
            if j in senalist and j in senblist:
                return True
    return False

def Fscore(evaluatedList, senOPscore):
    senNum = len(evaluatedList)
    FscoreList = []
    markList = []
    for i in range(senNum):
        FscoreList.append(1)
        markList.append(0)

    print("evaluatedList", evaluatedList)
    for i in range(senNum):
        if (markList[i]):
            continue
        markList, FscoreList = EvaSen(i, markList, evaluatedList, FscoreList, senOPscore)

    return FscoreList

def EvaSen(senA, markList, evaluatedList, FscoreList, senOPscore):
    print("senA", senA)
    print("markList", markList)
    print("FscoreList", FscoreList)
    for i in evaluatedList[senA]:
        if (not markList[i]):
            markList, FscoreList = EvaSen(i,markList,evaluatedList, FscoreList, senOPscore)
        FscoreList[senA] = FscoreList[senA] + FscoreList[i]*senOPscore[i]
    markList[senA] = 1
    return markList, FscoreList

def EvaluatedList(evaluationList):
    #print("evaluationList", evaluationList)
    senNum = len(evaluationList)
    #print(senNum)
    evaluatedList = []
    for i in range(senNum):
        evaluatedList.append([])
    for i in range(senNum):
        for ev in evaluationList[i]:
            #print("ev", ev)
            if ev != i:
                evaluatedList[ev].append(i)
    #print("evaluatedList", evaluatedList)
    return evaluatedList

def EvaluationBehaviorCapture(datasetName):
    sheetNameList, dataList = DataInput(datasetName)
    dataList = EvaluationBehaviorCap(dataList, sheetNameList, 0.1)
    wirteDataList(dataList, sheetNameList, datasetName)
