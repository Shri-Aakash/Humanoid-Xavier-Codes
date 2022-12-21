import face_recognition
# import dlib
#import pyrealsense2
import numpy as np
import pandas as pd
import os
import cv2

class Face_Recognition():
    df=pd.read_csv('Data.csv')
    encodeList=[]
    def __init__(self):
        for idx, encoding in enumerate(Face_Recognition.df['Encodings']):
            encoding = encoding.split(',')
            encoding = np.array(encoding)
            encoding = encoding.astype(float)
            Face_Recognition.encodeList.append(encoding)

    @staticmethod
    def findFace(encodeVal):
        dist = face_recognition.face_distance(Face_Recognition.encodeList, encodeVal)
        names = Face_Recognition.df['Name']
        idx = np.argmin(dist)
        if dist[idx] > 0.4:
            return 'Un-identified Face'
        else:
            if names[idx]:
                return names[idx]
            else:
                return 'Name Unavailable'

    @staticmethod
    def getFaces(img):
        faces=face_recognition.face_locations(img)
        encodeImg=face_recognition.face_encodings(img,faces)
        return (faces,encodeImg)

class Camera():
	def __init__(self):
		self.index=0
		self.arr=[]
		self.i=10
		while self.i > 0:
			cap=cv2.VideoCapture(self.index)
			if cap.read()[0]:
				self.arr.append(self.index)
				cap.release()
			self.index+=1
			self.i-=1
		print(self.arr)
		self.cap=cv2.VideoCapture(2)
		self.FPS=self.cap.get(cv2.CAP_PROP_FPS)
		print(f"Initizalied Camera index {self.arr[-1]} at {self.FPS}")

	def getFrame(self):
		ret,img=self.cap.read()
		return img

	def release(self):
		self.cap.release()


if __name__=='__main__':
	camera=Camera()
	fr=Face_Recognition()
	while True:
		frame=camera.getFrame()
		faces,encodeImg=fr.getFaces(frame)
		print(len(faces))
		for eF,fL in zip(encodeImg,faces):
			name=fr.findFace(eF)
			cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
		cv2.imshow('Face Recognition',frame)
		if cv2.waitKey(1) & 0xFF==ord('q'):
			break
	camera.release()