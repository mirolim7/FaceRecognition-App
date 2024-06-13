import openpyxl
import pandas as pd
import xlsxwriter
import os
import subprocess
from tkinter import filedialog
import tkinter as tk
import streamlit as st
from PIL import Image

path = 'images/'
name = 'barak-obama'

uploaded_files = st.file_uploader("Choose pictures", type=['png', 'jpg'], accept_multiple_files=True)
for i in range(len(uploaded_files)):
    #st.write(uploaded_file)
    bytes_data = uploaded_files[i].read()  # read the content of the file in binary
    #print(uploaded_files[i].name)#, bytes_data)
    #print(uploaded_files[i])
    with open(os.path.join(path + name + '/', uploaded_files[i].name), "wb") as f:
       f.write(bytes_data)  # write this content elsewhere
    
    
#    bytes_data = uploaded_file.read()
#    st.write("filename:", uploaded_file.name)
#    st.write(bytes_data)

########################## FROM web_str.py

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

st.title("Face Recognition App")

frame_placeholder = st.empty()

#button_field = st.empty()
start_btn_pressed = st.button('Start', key='start_button')
stop_btn_pressed = st.button('Stop', key='stop_button')

faces_detected_list = st.empty()

faces_detected = []


if start_btn_pressed:
    cap = cv2.VideoCapture(0)
    while cap.isOpened and not stop_btn_pressed:
        
        ret, frame = cap.read()
        img = cv2.flip(frame, 1)
        
        #if img is None:
         #   continue
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
        for encodedFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            name = pred_img(encodedFace)
            
            y1,x2,y2,x1 = faceLoc
            #st.write(faceLoc)
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
            #cv2.imshow('video', img)
    
        frame_placeholder.image(img, channels="RGB")
        time.sleep(0.01)

        if not ret:
            st.write('The video capture ended')
            break
    
        #st.write(name)
    
    cap.release() 
    cv2.destroyAllWindows()
    
#faces_detected_list.write('Faces detected: ')
#faces_detected_list.write(faces_detected)

    