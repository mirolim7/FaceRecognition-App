import cv2
import numpy as np
import os
import face_recognition
from image_ret_proc import *
import pickle

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    """faces = faceCascade.detectMultiScale(gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )"""
    
    facesCurFrame = face_recognition.face_locations(gray)
    encodesCurFrame = face_recognition.face_encodings(gray, facesCurFrame)
 
    for encodedFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        name = pred_img(encodedFace)
        
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
        cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
    cv2.imshow('video', img)
    
    k = cv2.waitKey(30)
    if k == 27:
        break
    
print(name)
    
cap.release()
cv2.destroyAllWindows()