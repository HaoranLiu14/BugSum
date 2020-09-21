from DataPreprocess import DataInput, SigDataInput, DataPreprocess, BugSumSigDataInput, BugSumSigDataWithoutTestInput
from EvaluationBehaviorCapture import EvaluationBehaviorCapture
from Util import DataListList2float, DataListList2int, StrList2FloatList, StrList2IntList, DataListListProcess, wirteDataList, wordResCounter, AccuracyMeasure, answerTypeTrans
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def BugSum(datasetName, beamSize):
    #DataPreprocess(datasetName)
    #EvaluationBehaviorCapture(datasetName)
    sheetNameList, dataList = DataInput(datasetName)
    sheetNum = len(sheetNameList)
    averageAcc = 0
    averageRecall = 0
    for i in range(sheetNum):
        print("Processing", sheetNameList[i])
        senList, lemSenList, senNumberList, authorList, senInvolvecd, senVecList, senOPscore, tfidfWordList, tfidfScoreList, ignoreList, goldenList, evaluationList, evaluationTimeList, buildInfoMark, fscoreList = BugSumSigDataInput(dataList, i)
        wordRes = wordResCounter(ignoreList, 0.25)
        print("wordRes", wordRes)
        #answer, loss = BeamSearchBert(senVecList, fscoreList, ignoreList, wordRes, senList, beamSize)
        answer, loss = BeamSearchBert(senVecList, fscoreList, ignoreList, buildInfoMark, wordRes, senList, beamSize)
        acc, recall = AccuracyMeasure(answer, goldenList, wordRes)
        transedAnswer = answerTypeTrans(answer, senList)
        dataList[i]["BugSumSelected"] = transedAnswer
        dataList[i]["Performance"] = [acc, recall].copy()
        averageAcc = averageAcc + acc
        averageRecall = averageRecall + recall
        print(acc, recall)
    wirteDataList(dataList, sheetNameList, datasetName)
    averageAcc = averageAcc/sheetNum
    averageRecall = averageRecall/sheetNum
    print("average:", averageAcc, averageRecall)
    return averageAcc, averageRecall


def BugSumWithoutTest(datasetName, beamSize):
    #DataPreprocess(datasetName)
    #EvaluationBehaviorCapture(datasetName)
    sheetNameList, dataList = DataInput(datasetName)
    sheetNum = len(sheetNameList)

    for i in range(sheetNum):
        print("Processing", sheetNameList[i])
        senList, lemSenList, senNumberList, authorList, senInvolvecd, senVecList, senOPscore, tfidfWordList, tfidfScoreList, ignoreList, evaluationList, evaluationTimeList, buildInfoMark, fscoreList = BugSumSigDataWithoutTestInput(dataList, i)
        wordRes = wordResCounter(ignoreList, 0.25)
        print("wordRes", wordRes)
        #answer, loss = BeamSearchBert(senVecList, fscoreList, ignoreList, wordRes, senList, beamSize)
        answer, loss = BeamSearchBert(senVecList, fscoreList, ignoreList, buildInfoMark, wordRes, senList, beamSize)
        print("selectedSen", answer)
        transedAnswer = answerTypeTrans(answer, senList)

        dataList[i]["BugSumSelected"] = transedAnswer
        #dataList[i]["Performance"] = [acc, recall].copy()
    wirteDataList(dataList, sheetNameList, datasetName)



def BeamSearchBert(senVecList, fscore, ignoreList, buildInfoMark, wordRes, senList, beamSize):
    print("ignoreList", ignoreList)

    senNum = len(senVecList)

    Lnew = []
    LnewLoss = []
    Lold = []
    Chosen = []
    ChosenLoss = []

    Nvec = []
    #Ovec = []

    imptlist = []

    for i in range(senNum):
        if (ignoreList[i] == 1):
            continue
        imptlist.clear()
        imptlist.append(i)
        Lold.append(imptlist.copy())
        #Ovec.append(zeroT.copy())

    print("senVecList", senVecList[0])
    WeightedDataVec = VecMulFscore(senVecList, fscore)

    FTVec = GenFullTVec(WeightedDataVec, ignoreList)

    while (len(Lold) > 0):
        # print("Lold", Lold)
        LoldNum = len(Lold)
        for candiListNumber in range(LoldNum):
            candiList = Lold[candiListNumber]
            # print("candiList", candiList)
            for i in range(senNum):
                if (ignoreList[i] == 1):
                    continue
                if i not in candiList:
                    newCandi = candiList.copy()
                    newCandi.append(i)
                    # print("newCandi", newCandi)
                    flag = 1
                    for kList in Lnew:
                        if (ListCmp(kList, newCandi)):
                            flag = 0
                            break
                    if (flag):
                        # if (SentenceNumLengthCheck(sentencelist, newCandi, 2*wordRes)):
                        #ReconDvec = ReconsDvec(Svecs_SVM, newCandi, FcoreList, MODELD)
                        ReconFTvec = ReconFullTVec(senVecList, newCandi)

                        # ReconDvec = ReconDvec.cpu().numpy().tolist()[0]
                        # loss = CosineSimiarlty(ReconDvec, Dvec)
                        # print("Loss", loss)

                        loss = BertVecLoss(FTVec, ReconFTvec)
                        # print("start")
                        # infroincrease = criterion(ReconDvec, Dvec)
                        # Novelties = novelty(candiListNumber, Ovec, newCandi, SvecList, FcoreList, 0.2)
                        # Novelties = criterion()
                        # loss = infroincrease + Novelties
                        # loss = criterion(ReconDvec, Dvec) + novelty(candiListNumber, Ovec, newCandi, SvecList, FcoreList, 0.1)
                        # print("end")

                        # Lnew, LnewLoss, Chosen, ChosenLoss = appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, sentencelist, wordRes, beamSize)
                        Lnew, LnewLoss, Chosen, ChosenLoss, Nvec = appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, senList, wordRes, beamSize, Nvec, ReconFTvec)
            '''
            print("Lnew", Lnew)
            print("LnewLoss", LnewLoss)
            print("Lold", Lold)
            print("Chosen", Chosen)
            print("ChosenLoss", ChosenLoss)
            '''
        # print("Lnew", Lnew)
        # print("LnewLoss", LnewLoss)
        # print("Lold", Lold)
        Lold = Lnew.copy()
        Ovec = Nvec.copy()
        Lnew.clear()
        LnewLoss.clear()
    answer = []
    loss = 0
    if (wordRes == 1):
        for i in range(senNum):
            if (ignoreList[i] == 1):
                continue
            answer.append(i)
    else:
        print("ChosenLossSize", len(Chosen), wordRes)
        pos, loss = locateSamllestOne(ChosenLoss)
        answer = Chosen[pos]

    # print(pos, loss)
    return answer, loss
'''
def wordResCounter(ignoreList, goldenList, percent, goldenPercent):
    senNum = len(ignoreList)
    goldenSenNum = len(goldenList)
    counter = 0
    for i in range(senNum):
        if (ignoreList[i]==0):
            counter = counter + 1
    answer = int(counter*percent)
    if answer == 0:
        answer = 1

    goldenCounter = 0
    for i in range(goldenSenNum):
        if (goldenList[i]==1):
            goldenCounter = goldenCounter + 1
    goldenAnswer = int(goldenCounter*goldenPercent)
    if goldenAnswer>counter:
        goldenAnswer = counter
    if goldenAnswer == 0:
        goldenAnswer = 1
    return answer, goldenAnswer
'''




def BertVecLoss(Veca, Vecb):
    #print("Veca", len(Veca))
    #print("Vecb", len(Vecb))
    VecLen = len(Veca)
    lossAnswer = 0.0
    for i in range(VecLen):
        lossAnswer = lossAnswer + (Veca[i]-Vecb[i])**2
    return lossAnswer

def VecAdd(Veca, Vecb):
    VecLen = len(Veca)
    answerList = []
    for i in range(VecLen):
        answerList.append(Veca[i]+Vecb[i])
    return answerList

def VecMulNumber(Veca, number):
    VecLen = len(Veca)
    answerList = []
    #print("Veca", Veca)
    for i in range(VecLen):
        #print("Veca[i]", Veca[i])
        answerList.append(Veca[i]*number)
    return answerList

def VecMulFscore(senVecList, senOPscore):
    senNum = len(senVecList)
    newSenVecList = []
    for i in range(senNum):
        #print("senVecList[i]", senVecList[i][0])
        imptList = VecMulNumber(senVecList[i], senOPscore[i])
        newSenVecList.append(imptList.copy())
    return newSenVecList

def GenFullTVec(senVecList, buildInfoMark):
    vecLen = len(senVecList[0])
    FullTVec = []
    for i in range(vecLen):
        FullTVec.append(0)
    senVecListNum = len(senVecList)
    for inum in range(senVecListNum):
        i=senVecList[inum]
        if (buildInfoMark[inum]!=1):
            FullTVec = VecAdd(FullTVec, i)
    return FullTVec

def ReconFullTVec(senVecList, chosenList):
    FullTVec = []
    vecLen = len(senVecList[0])
    for i in range(vecLen):
        FullTVec.append(0)
    for i in chosenList:
        FullTVec = VecAdd(FullTVec, senVecList[i])
    return FullTVec





def BeamSearch(sentencelist, avoidSentenceList, FcoreList, Encoder, beamSize, wordRes, MODELD):
    #print("avoidSentenceList", avoidSentenceList)
    #print("wordRes", wordRes)
    with torch.no_grad():#这个是取消了反向传播
        criterion = torch.nn.MSELoss()
        #criterion = CosineSimiarlty()
        sennum = len(sentencelist)
        Lnew = []
        LnewLoss = []
        Lold = []
        Chosen = []
        ChosenLoss =[]

        Nvec = []
        Ovec = []

        imptlist = []
        zeroT = zeroTensor(2 * MODELD)

        for i in range(sennum):
            if (avoidSentenceList[i]==1):
                continue
            imptlist.clear()
            imptlist.append(i)
            Lold.append(imptlist.copy())
            Ovec.append(zeroT.copy())

        #print("sentencelist", sentencelist)

        Svecs_SVM, SvecList = AverageDvecGenProcess(sentencelist, FcoreList, sennum, Encoder, MODELD)

        #chosenlist = selectAll(sennum)

        chosenlist = reverseAvoid(avoidSentenceList)

        #Dvec = ReconsDvec(Svecs_SVM, chosenlist, FcoreList)
        Dvec = DvecGen(Svecs_SVM, chosenlist, FcoreList, MODELD)
        #Dvec = Dvec.cpu().numpy().tolist()[0]

        while(len(Lold)>0):
            #print("Lold", Lold)
            LoldNum = len(Lold)
            for candiListNumber in range(LoldNum):
                candiList = Lold[candiListNumber]
                #print("candiList", candiList)
                for i in range(sennum):
                    if (avoidSentenceList[i]==1):
                        continue
                    if i not in candiList:
                        newCandi = candiList.copy()
                        newCandi.append(i)
                        #print("newCandi", newCandi)
                        flag = 1
                        for kList in Lnew:
                            if (ListCmp(kList, newCandi)):
                                flag = 0
                                break
                        if (flag):
                            #if (SentenceNumLengthCheck(sentencelist, newCandi, 2*wordRes)):
                            ReconDvec = ReconsDvec(Svecs_SVM, newCandi, FcoreList, MODELD)

                            #ReconDvec = ReconDvec.cpu().numpy().tolist()[0]
                            #loss = CosineSimiarlty(ReconDvec, Dvec)
                            #print("Loss", loss)

                            loss = criterion(ReconDvec, Dvec)
                            #print("start")
                            #infroincrease = criterion(ReconDvec, Dvec)
                            #Novelties = novelty(candiListNumber, Ovec, newCandi, SvecList, FcoreList, 0.2)
                            #Novelties = criterion()
                            #loss = infroincrease + Novelties
                            #loss = criterion(ReconDvec, Dvec) + novelty(candiListNumber, Ovec, newCandi, SvecList, FcoreList, 0.1)
                            #print("end")

                            #Lnew, LnewLoss, Chosen, ChosenLoss = appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, sentencelist, wordRes, beamSize)
                            Lnew, LnewLoss, Chosen, ChosenLoss, Nvec = appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, sentencelist, wordRes, beamSize, Nvec, ReconDvec)
                #print("Lnew", Lnew)
                #print("LnewLoss", LnewLoss)
                #print("Lold", Lold)
                #print("Chosen", Chosen)
                #print("ChosenLoss", ChosenLoss)
            #print("Lnew", Lnew)
            #print("LnewLoss", LnewLoss)
            #print("Lold", Lold)
            Lold = Lnew.copy()
            Ovec = Nvec.copy()
            Lnew.clear()
            LnewLoss.clear()
        answer = []
        loss = 0
        if(wordRes == 1):
            for i in range(sennum):
                if (avoidSentenceList[i] == 1):
                    continue
                answer.append(i)
        else:
            print("ChosenLossSize", len(Chosen), wordRes)
            pos, loss = locateSamllestOne(ChosenLoss)
            answer = Chosen[pos]

        #print(pos, loss)
        return answer, loss

def reverseAvoid(senList):
    senNum = len(senList)
    newSenList = []
    for i in range(senNum):
        if senList[i] == 0:
            newSenList.append(i)
    return newSenList

def DvecGen(Svecs_SVM, chosenlist, intsvm, MODELD):
    Dvec = torch.zeros((1, 2*MODELD))
    Dvec = Dvec.to(device)
    for num in chosenlist:
        Dvec = torch.add(Dvec, Svecs_SVM[num])

    '''total = 0
    for i in chosenlist:
        total = total + intsvm[i]'''

    #average_sennum = 1.0/total

    #average_sennum = 1.0/len(chosenlist)

    average_sennum = 1.0

    average_sennum = torch.tensor(average_sennum, dtype=torch.float, device=device)
    average_sennum = average_sennum.to(device)
    Dvec = torch.mul(Dvec, average_sennum)
    return Dvec

def ReconsDvec(Svecs_SVM, chosenlist, intsvm, MODELD):
    Dvec = torch.zeros((1, 2*MODELD))
    Dvec = Dvec.to(device)
    for num in chosenlist:
        Dvec = torch.add(Dvec, Svecs_SVM[num])

    '''total = 0
    for i in chosenlist:
        total = total + intsvm[i]

    average_sennum = 1.0/total'''

    average_sennum = 1.0/len(chosenlist)

    average_sennum = 1.0

    average_sennum = torch.tensor(average_sennum, dtype=torch.float, device=device)
    average_sennum = average_sennum.to(device)
    Dvec = torch.mul(Dvec, average_sennum)
    return Dvec

def ListCmp(list1, list2):
    if (len(list1)!=len(list2)):
        return False
    for itemk in list1:
        if itemk not in list2:
            return False
    return True

def locateSamllestOne(List):
    smallestNumber = 100
    smallestPos = 0
    listLength = len(List)
    for i in range(listLength):
        if List[i] < smallestNumber:
            smallestNumber = List[i]
            smallestPos = i
    return smallestPos, smallestNumber

def zeroTensor(vecSize):
    TenVec = []
    for i in range(vecSize):
        TenVec.append(0)

    #answer = torch.tensor(TenVec, dtype=torch.float, device=device)
    answer = TenVec
    return answer

def appendLnew(newCandi, loss, Lnew, LnewLoss, Chosen, ChosenLoss, sentencelist, wordNumRes, beamSize, Nvec, ReconVec):
    #print("Lnew", Lnew)
    #print("LnewLoss", LnewLoss)
    #print("newCandi", newCandi, loss)
    if loss<0:
        loss = loss*-1

    #ReconVec = ReconVec.cpu().numpy().tolist()[0]

    if extendable(sentencelist, newCandi, wordNumRes):
        if len(Lnew) < beamSize:
            Lnew.append(newCandi.copy())
            LnewLoss.append(loss)
            Nvec.append(ReconVec.copy())
        else:
            pos, bigestLossNumber = locateBigOne(LnewLoss)
            #print("pos", pos)
            # print("bigestLossNumber", bigestLossNumber)
            # if bigestLossNumber > loss or (bigestLossNumber == loss and (SentenceNumLengthCount(sentencelist, Lnew[pos]) > SentenceNumLengthCount(sentencelist, newCandi))):
            if bigestLossNumber > loss:
                # print("replacing", pos)
                LnewLoss[pos] = loss
                Lnew[pos] = newCandi.copy()
                Nvec[pos] = ReconVec.copy()
    else:
        if (len(Chosen) < beamSize):
            Chosen.append(newCandi.copy())
            ChosenLoss.append(loss)

        else:
            pos, bigestLossNumber = locateBigOne(ChosenLoss)
            # if bigestLossNumber > loss or (bigestLossNumber == loss and (SentenceNumLengthCount(sentencelist, Chosen[pos]) > SentenceNumLengthCount(sentencelist, newCandi))):
            if bigestLossNumber > loss:
                ChosenLoss[pos] = loss
                Chosen[pos] = newCandi.copy()
    return Lnew, LnewLoss, Chosen, ChosenLoss, Nvec

def extendable(sentenceList, imptList, TargetSenNum):
    if (len(imptList)+1 < TargetSenNum):
        return True
    else:
        return False

def locateBigOne(List):
    #print("List", List)
    bigestNumber = 0
    bigestPos = 0
    listLength = len(List)
    #print("Len", listLength)
    for i in range(listLength):
        if List[i] > bigestNumber:
            bigestNumber = List[i]
            bigestPos = i
    return bigestPos, bigestNumber

def locateSamllestOne(List):
    smallestNumber = 100
    smallestPos = 0
    listLength = len(List)
    for i in range(listLength):
        if List[i] < smallestNumber:
            smallestNumber = List[i]
            smallestPos = i
    return smallestPos, smallestNumber

def AverageDvecGenProcess(intcom, intsvm, sennum, Encoder, MODELD):
    Svecs = []
    targetvecs = []
    SvecList = []
    for senk in range(sennum):
        input_tensor = torch.tensor(intcom[senk].copy(), dtype=torch.long, device=device)
        input_tensor = input_tensor.to(device)
        Encoder = Encoder.to(device)
        encodeHidden = ProcSgen(Encoder, input_tensor)

        CombinedVec = combineForthAndBack(encodeHidden)

        Svecs.append(CombinedVec.clone().detach())
        targetvecs.append(CombinedVec.clone().detach())
        #SvecList.append(CombinedVec.clone().detach().cpu().numpy().tolist().copy())

    #print("length", len(intcom), len(intsvm))
    #print(intsvm)

    SVMvec = torch.tensor(intsvm, dtype=torch.float, device=device)
    #SVMvec = torch.softmax(SVMvec, 0)
    SVMvec = SVMvec.view(-1, 1).repeat(1, 2*MODELD)
    SVMvec = SVMvec.to(device)

    Svecstensor = torch.cat(Svecs)

    #print("Svecs", Svecs[0])

    #print("Size", Svecstensor.size())
    #print("Before Svecstensor", Svecstensor)


    Svecs_SVM = torch.mul(Svecstensor, SVMvec)
    Svecs_SVM = Svecs_SVM.detach()

    for i in Svecs_SVM:
        SvecList.append(i.clone().detach().cpu().numpy().tolist().copy())

    Svecs_SVM = Svecs_SVM.to(device)

    #print("Svecs_SVM", Svecs_SVM)

    return Svecs_SVM, SvecList


def ProcSgen(encoder, input_tensor):#inputtensor是一句话
    input_length = input_tensor.size(0)
    encoder_hidden = encoder.initHidden()
    encoder_hidden = encoder_hidden.to(device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    #return encoder_hidden[0].clone().detach()#Bidirection hidden, only take forward
    return encoder_hidden.clone().detach()  # Bidirection hidden, take them both

def combineForthAndBack(encodeHidden):
    forthVec = encodeHidden[0]#[1,400]
    backVec = encodeHidden[1]#[1,400]
    answer = torch.cat((forthVec, backVec), 1)#[1,800]
    #print("answer", answer.size())
    return answer
