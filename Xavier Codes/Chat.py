import random
import json
import torch
from model import NeuralNet
from nltkUtils import tokenize,stem,bagOfWords
from FaceRecognition_NormalCam import Camera,Face_Recognition
import cv2
name =''

class ChatBot():
    with open('intents.json','r') as f:
        intents=json.load(f)
    FILE='model2.pth'
    data=torch.load(FILE)
    botName='Alpha'
    def __init__(self):
        #c1=Camera()
        #fr=Face_Recognition()
        self.input_size = ChatBot.data["input_size"]
        self.hidden_size = ChatBot.data["hidden_size"]
        self.output_size = ChatBot.data["output_size"]
        self.all_words = ChatBot.data['all_words']
        self.tags = ChatBot.data['tags']
        self.model_state = ChatBot.data["modelState"]
        self.model=NeuralNet(self.input_size,self.hidden_size,self.output_size)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

    def Chat(self,sentence,c1,fr):
        self.sentence=tokenize(sentence)
        self.x=bagOfWords(self.sentence,self.all_words)
        self.x=self.x.reshape(1,self.x.shape[0])
        self.x=torch.from_numpy(self.x)
        self.output=self.model(self.x)
        self._,self.pred=torch.max(self.output,dim=1)
        self.tag = self.tags[self.pred.item()]
        self.probs = torch.softmax(self.output, dim=1)
        self.prob = self.probs[0][self.pred.item()]
        if self.prob.item() > 0.75:
            if self.tag=='goodbye':
                return 'exit'
            elif self.tag == "follow me" or self.tag == "items":
                global name
                frame=c1.getFrame()
                faces,encodeImg=fr.getFaces(frame)
                if len(faces):
                    for eF,fL in zip(encodeImg,faces):
                        name=fr.findFace(eF)
                        cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv2.imshow('Face Recognition',frame)
                    if name=='Un-identified Face':
                        if self.tag == "follow me":
                            print(f"{ChatBot.botName}: Sorry your face is not in the database and thus i can't follow you.")
                        else:
                            print(f"{ChatBot.botName}: Sorry your this feature is only for faculty.")
                    else:
                        for intent in ChatBot.intents['intents']:
                            if self.tag == intent['tag']:
                                print(f"{ChatBot.botName}: {name} show me your QR")
                else:
                    print("Your Face is not visible.Please Keep your face within the frame")
            else:
                for intent in ChatBot.intents['intents']:
                    if self.tag == intent["tag"]:
                        print(f"{ChatBot.botName}: {random.choice(intent['responses'])}")
        else:
            print(f"{ChatBot.botName}: I do not understand...")
        return ()

# with open('intents.json','r') as f:
#     intents=json.load(f)

# FILE='model2.pth'
# data=torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["modelState"]

# model=NeuralNet(input_size,hidden_size,output_size)
# model.load_state_dict(model_state)
# model.eval()

# botName="Alpha"
# print("Let's chat! type 'quit' to exit")
# while True:
#     sentence=input("User: ")
#     if sentence=='quit':
#         break
#     sentence=tokenize(sentence)
#     x=bagOfWords(sentence,all_words)
#     x=x.reshape(1,x.shape[0])
#     x=torch.from_numpy(x)

#     output=model(x)
#     _,pred=torch.max(output,dim=1)

#     tag = tags[pred.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][pred.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{botName}: {random.choice(intent['responses'])}")

#     else:
#         print(f"{botName}: I do not understand...")

if __name__=='__main__':
    cb=ChatBot()
    while True:
        sentence=input("User: ")
        res=cb.Chat(sentence)
        if res=='exit':
            break
