'''Import Required Libraries'''
import face_recognition
# import dlib
import pyrealsense2
import numpy as np
import pandas as pd
import os
import cv2
from realsense_depth import *

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
        if dist[idx] > 0.6:
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



if __name__=='__main__':
    dc=DepthCamera()
    fr=Face_Recognition()
    while True:
        ret,depthFrame,colorFrame=dc.get_frame()
        faces,encodeImg=fr.getFaces(colorFrame)
        for eF,fL in zip(encodeImg,faces):
            name=fr.findFace(eF)
            cv2.putText(colorFrame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('Face Recognition',colorFrame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    dc.release()