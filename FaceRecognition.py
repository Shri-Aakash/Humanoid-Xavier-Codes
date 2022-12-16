'''Import Required Libraries'''
import face_recognition
# import dlib
import cv2
import numpy as np
import pandas as pd
import os

cap = cv2.VideoCapture(0)

df = pd.read_csv('Data.csv')
#################################################################################
encodeList = []
for idx, encoding in enumerate(df['Encodings']):
    encoding = encoding.split(',')
    encoding = np.array(encoding)
    encoding = encoding.astype(float)
    encodeList.append(encoding)


######################################################################################
def findFace(encodeVal):
    dist = face_recognition.face_distance(encodeList, encodeVal)
    names = df['Name']
    idx = np.argmin(dist)
    if dist[idx] > 0.6:
        return 'Un-identified Face'
    else:
        if names[idx]:
            return names[idx]
        else:
            return 'Name Unavailable'
