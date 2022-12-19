import pyrealsense2
import cv2
#import numpy as np
from realsense_depth import *


cascadeClassifierPath=r'D:\Environments\ltts\Haar Cascade\haarcascade_frontalface_default.xml'
faceCascade=cv2.CascadeClassifier(cascadeClassifierPath)
dc=DepthCamera()

while True:
    ret,depthFrame,colorFrame=dc.get_frame()
    try:
        faces=faceCascade.detectMultiScale(colorFrame,1.1,5)
        x,y,w,h=faces[0]
        points=[x+w/2,y+h/2]
        distance=depthFrame[int(points[1]),int(points[0])]
        cv2.rectangle(colorFrame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(colorFrame,(int(x+w/2),int(y+h/2)),4,3)
        cv2.putText(colorFrame,f'{distance/10}cm',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    except:
        pass
    cv2.imshow("Color Frame", colorFrame)
    print(distance)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break