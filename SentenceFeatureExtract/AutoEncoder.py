from __future__ import unicode_literals, print_function, division
from io import open

import random

import torch
import torch.nn as nn
from torch import optim
from Util import readFileList
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EncoderRNN,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size,hidden_size)
        self.gru = torch.nn.GRU(hidden_size,hidden_size,bidirectional=True)

    def forward(self,input,hidden):
        #print("INPUT",input.size(),input)
        embeded = self.embedding(input).view(1,1,-1)
        output = embeded
        output,hidden = self.gru(output,hidden)
        return output,hidden

    def initHidden(self):
        return torch.zeros(2,1,self.hidden_size,device=device)

class DecoderRNN(torch.nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(output_size,hidden_size)
        self.gru = torch.nn.GRU(hidden_size,hidden_size,bidirectional=True)
        self.out = torch.nn.Linear(2*hidden_size,output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        #print("DecoderSize", input.size(), hidden.size())
        output = self.embedding(input).view(1,1,-1)
        output = torch.relu(output)
        #print("OUTPUT",output.size())
        output,hidden = self.gru(output, hidden)
        #print("Hidden", hidden.size())
        output = self.softmax(self.out(output[0]))
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,criterion,max_length = MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output,encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0,0][:WordvectDimention]

    decoder_input = torch.tensor([[SOS_token]],device=device)
    #decoder_input = decoder_input.cuda()
    decoder_input = decoder_input.to(device)

    #decoder_hidden = encoder_hidden[0].unsqueeze(0)
    decoder_hidden = encoder_hidden
    #print("HiddenSize", encoder_hidden.size())
    decoder_hidden = decoder_hidden.to(device)

    use_tencher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_tencher_forcing:
        for di in range(target_length):
            #print("shit?")
            #decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden[0].unsqueeze(0))
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)  # 概率 和 one-hot
            #print(topi.item(), "and", target_tensor[di].item())
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            decoder_input = target_tensor[di] #teacher forcing
    else:
        for di in range(target_length):
            #decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden[0].unsqueeze(0))
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv,topi = decoder_output.topk(1) #概率 和 one-hot
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output,target_tensor[di].unsqueeze(0))
            if decoder_input.item() == EOS_token:
                break

    #print(loss)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def saveAllmodel(DataSetName, name, encoder1, decoder1):
    SPATH = "Model\\{}\\Sgenerator".format(DataSetName) + name
    EPATH = "Model\\{}\\DecoderRNN".format(DataSetName) + name
    torch.save(encoder1, SPATH)
    torch.save(decoder1, EPATH)

def prepareData(FileHead, Filenumber):
    filename = FileHead + str(Filenumber) + ".txt"
    lines = open(filename, encoding='utf-8').read().split('\n')

    intsen = []
    intsvclen = []

    lineNum = len(lines)
    for linenumber in range(lineNum):
        if linenumber%2==0:
            continue
        line = lines[linenumber]
        line = line.split()
        imptline = []
        for sig in line:
            imptline.append(int(sig))
        if (len(imptline)==0):
            continue
        intsen.append(imptline.copy())
        intsvclen.append(len(imptline))

    sennum = len(intsen)

    return intsen, intsvclen, sennum

#def trainIters(encoder, decoder, n_iters, FileHead, FileNumm, learning_rate =0.01):
def trainIters(encoder, decoder, n_iters, DatasetName, learning_rate=0.01):
    FileHead = "Vec\\{}\\Vec".format(DatasetName)
    FileNList = readFileList(DatasetName)
    FN = len(FileNList)
    encoder_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(),lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter in range(1,n_iters + 1):
        try:
            #num = random.randint(1, FileNumm + 1)
            #num = random.randint(0, FN-1)
            for num in FileNList:
            #num = FileNList[num]
                intcom, intsveclen, sennum = prepareData(FileHead, num)
                print("intcom", intcom)
                targetcom = intcom.copy()
                a_loss = 0

                for senk in range(sennum):
                    input_tensor = torch.tensor(intcom[senk][:intsveclen[senk]].copy(), dtype=torch.long,device=device)
                    target_tensor = torch.tensor(targetcom[senk][:intsveclen[senk]].copy(), dtype=torch.long,device=device)

                    input_tensor = input_tensor.to(device)
                    target_tensor = target_tensor.to(device)

                    loss = train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
                    a_loss += loss
                print(iter, num, a_loss/sennum)
        except FileNotFoundError:
            continue
    return encoder, decoder

#WordvectDimentionList = [1,2,5,7,10,20,50,100,200,300,400,500,600,700,800,900,1000]

SOS_token = 0
EOS_token = 1

##filter
MAX_LENGTH = 30#一句话的最大长度

teacher_forcing_ratio = 0.5

def trainEDcoder(WordvectDimentionList, DName):
    if (not os.path.exists("Model\\{}".format(DName))):
        os.makedirs("Model\\{}".format(DName))
    for k in WordvectDimentionList:
        print("Processing K:", k)
        WordvectDimention = k
        # for 主观判断
        hidden_size = WordvectDimention
        encoder1 = EncoderRNN(2000, hidden_size).to(device)
        decoder1 = DecoderRNN(hidden_size, 2000).to(device)
        encoder1, decoder1 = trainIters(encoder1, decoder1, 50, DName)
        saveAllmodel(DName, str(WordvectDimention), encoder1, decoder1)