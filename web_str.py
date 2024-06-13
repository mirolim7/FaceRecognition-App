import cv2
import streamlit as st
import numpy as np
import pandas as pd
import os
from image_ret_proc import *
import time
from login_form import *
from admin_roles import *


def show_and_detect():
    
    faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
    
    st.title("Face Recognition App")

    frame_placeholder = st.empty()
    
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


if 'show_imgs_btn' not in st.session_state:
    st.session_state['show_imgs_btn'] = False

def click_show_imgs_btn():
    st.session_state['show_imgs_btn'] = True
    

def check_validity(new_username, new_password):
    return ((len(new_username) > 3) and (len(new_password) > 5) and check_user_name(new_username))

if 'username' not in st.session_state:
    st.session_state['username'] = None

if st.session_state['username'] != None:
    
    col1, col2, col3 = st.columns([2,1,1])
    with col2:
        st.write(st.session_state['username'])
    
    with col3:
        logout_btn = st.button('Log out')
        if logout_btn:
            st.session_state['username'] = None
        
        
    if st.session_state['username'] == 'admin':
        section = st.selectbox('Choose a section', ('Users', 'Persons', 'Face Recognition App'))
        
        if section == 'Users':
            users = show_users()
            #st.write(users)
            for i, r in users.iterrows():
               u_col1, u_col2, u_col3 = st.columns([1,1,1])
               with u_col1:
                   st.write('Username: ' + r['username'] + '    Role: ' + r['role'])
                   
               with u_col2:
                   rem_btn = st.button('Remove user', key=f'rem_user_btn_{i}')
                   #st.write(users[i])
                   if rem_btn:
                       remove_user(r['username'])
                   
            #, on_click=click_btn, args=['add_user_btn'])
            
            #if st.session_state.add_user_btn:
                
                #st.session_state['new_user'] = None
                
            new_username = st.text_input('New User Name', key='add_user_name')
            new_password = st.text_input('New User Password', type='password', key='add_user_pass')
            
            add_user_btn = st.button('Add new user', key='add_user_btn')
            
                #add_user_submit_btn = st.button('Add user', key='add_user_submit_btn')
                
            if add_user_btn:
                
                #if add_user_btn:# and st.session_state['new_user']:       cheto zdes ne to
                
                #st.write(check_user_name(new_username))
                
                #st.write(check_validity(new_username, new_password))
                
                if check_validity(new_username, new_password):
                        #st.write('passed')
                    add_user(new_username, new_password)
                    st.success('Added new user!')
                        
                else:
                    st.warning('Enter correct username and password')
                        
                        
        
        elif section == 'Persons':
            
            retrain_model_btn = st.button('Retrain faces', key='retrain_model_btn')
            
            if retrain_model_btn:
                if update_face_encodings():
                    st.success('Images encoded')
            
            all_persons = all_persons()
            
            persons_imgs_cnt = 0
            
            for i in range(0, len(all_persons)):
                p_col1, p_col2, p_col3, p_col4 = st.columns([1,2,3,1])
                
                with p_col1:
                    st.write(all_persons[i].replace('-', ' ').upper())
                    
                with p_col2:    
                    show_imgs_btn = st.button('Show Images', key=f'show_imgs_{i}', on_click=click_show_imgs_btn)
                    if st.session_state['show_imgs_btn']:
                        person_imgs = person_images(all_persons[i])
                        for j in range(0, len(person_imgs)):
                            
                            #show_image(person_imgs[j])
                            st_img_col, rem_img_btn = st.columns([1,1]) 
                            image = Image.open(person_imgs[j])
                            st.write("filename:", person_imgs[j])
                            with st_img_col:
                                st.image(image, width=100)
                                
                            #with rem_img_btn:
                            rem_img_btn = st.button('Remove image', key=f'rem_img_btn_{persons_imgs_cnt}')
                            if rem_img_btn:
                                if remove_img(person_imgs[j]):
                                    
                                    st.warning('Image removed')
                        
                                    st.session_state['show_imgs_btn'] = False
                                    
                            
                            persons_imgs_cnt += 1
                
                with p_col3:
                    #add_imgs_btn = st.button('Add Images', key=f'add_imgs_{i}', on_click=click_add_imgs_btn)
                    #if st.session_state['add_imgs_btn']:
                        #cap_btn_col = st.columns([1])
                        
                        #with cap_btn_col:
                        #capture_btn = st.button('Capture and save', key=f'cap_btn_{i}')
                        #if capture_btn:
                                #capture_image(p)
                            #cam = cv2.VideoCapture(0)
                            #cv2.namedWindow("Capture image")
                            #img_counter = 0

                            #st.write('Press the SPACE key to shot a photo') 
                            #st.write('Press ESC to finish the capture')

                            #while True:
                            #    ret, frame = cam.read()
                                    
                            #    st.image(ret)
                                    
                            #    if not ret:
                            #        print("failed to grab frame")
                            #        break
                            #    cv2.imshow("test", frame)
                                    
                                    

                             #   k = cv2.waitKey(1)
                              #  if k%256 == 27:
                                    # ESC pressed
                               #     print("Escape hit, closing...")
                                #    break
                                #elif k%256 == 32:
                                        # SPACE pressed
                                 #   img_name = "images/{}/{}.jpg".format(p, p + '_' + img_counter)
                                  #  cv2.imwrite(img_name, frame)
                                    #print("{} written!".format(img_name))
                                   # img_counter += 1

                           # cam.release()
                            #cv2.destroyAllWindows()
                            #if img_counter > 0:
                             #   print('Images saved!')

                            #st.warning('Image was captured and saved')
                                
                                
                        uploaded_files = st.file_uploader("Choose images", type=['png', 'jpg'], accept_multiple_files=True, key=f'upl_imgs_{i}')
                        #st.write(all_persons[i])
                        if uploaded_files:
                            #st.write(uploaded_files)
                            
                            save_upl_btn = st.button('Save upload', key=f'save_upl_btn_{i}')
                                
                            if save_upl_btn:
                                #st.write('clicked')
                                #st.session_state['add_imgs_btn'] = False
                                add_images(uploaded_files, all_persons[i])
                            
                        
                                
                with p_col4:
                    rem_person_btn = st.button('Remove person', key=f'rem_person_btn_{i}')
                    if rem_person_btn:
                        remove_person(all_persons[i])
                        
                        
            new_person_name = st.text_input('New Person Name', key='add_person_name')
                
            add_person_btn = st.button('Add new person', key='add_person_btn')
            
            if add_person_btn:
                add_person(new_person_name)
                        
        
        elif section == 'Face Recognition App':
            show_and_detect()
        
        
    else:
        show_and_detect()
    

        
else:
    st.session_state['username'] = login_form()
    
