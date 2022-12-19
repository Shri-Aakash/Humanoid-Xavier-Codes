import json
import numpy as np
from nltkUtils import tokenize,stem,bagOfWords
from model import NeuralNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

with open('intents.json','r') as f:
    intents=json.load(f)

#print(intents)
all_words=[]
tags=[]
Xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        Xy.append((w,tag))

ignoreWords=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignoreWords]
all_words=sorted(set(all_words))
tags=sorted(set(tags))
#print(all_words,tags,sep='\n')
xTrain,yTrain=[],[]
for pattern,tag in Xy:
    bag=bagOfWords(pattern,all_words)
    xTrain.append(bag)

    label=tags.index(tag)
    yTrain.append(label)

xTrain=np.array(xTrain)
yTrain=np.array(yTrain)

class ChatDataset(Dataset):
    def __init__(self):
        self.nSamples=len(xTrain)
        self.xData=xTrain
        self.yData=yTrain

    def __getitem__(self,index):
        return self.xData[index],self.yData[index]

    def __len__(self):
        return self.nSamples

##Hyperparametres
batchSize=8
hiddenSize=8
outputSize=len(tags)
inputSize=len(xTrain[0])
learningRate=0.001
numEpochs=1000
#print(inputSize,len(all_words))
#print(outputSize,tags)


dataset=ChatDataset()
trainLoader=DataLoader(dataset=dataset,batch_size=batchSize,shuffle=True,num_workers=0)

device=torch.device('cuda')
model=NeuralNet(inputSize,hiddenSize,outputSize)

#loss and optimizer
criteria=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learningRate)

for epoch in range(numEpochs):
    for words,labels in trainLoader:
        words=words
        labels=labels.to(dtype=torch.long)

        outputs=model(words)
        loss=criteria(outputs,labels)

        #Backpropogation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%100==0:
        print(f'epoch {epoch+1}/{numEpochs},loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data={
"modelState":model.state_dict(),
"input_size":inputSize,
"hidden_size":hiddenSize,
"output_size":outputSize,
"all_words":all_words,
"tags":tags
}

file="model2.pth"
torch.save(data,file)


print(f'training complete. file saved to {file}')
