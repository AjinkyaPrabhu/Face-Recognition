import cv2
import numpy as np
import os
cap = cv2.VideoCapture(0)
face_cascades = cv2.CascadeClassifier('face.xml')
file_name = "training_set.npy"

if os.path.isfile(file_name):
    print("Loading Previously Saved File")
    training_data = list(np.load(file_name))
else:
    print("No Previosly Saved File .. Creating New File")
    training_data = []
is_main_person = 0



width=128
height=128


def isMainPerson(x):
    if x==1:
        return [1,0]
    else:
        return [0,1]


while True:
    ret,frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascades.detectMultiScale(gray,1.3,5)
    if len(faces)!=0:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cropped_face = frame[y:y+h,x:x+w]
        cv2.imshow('frame',frame)
        cv2.imshow('Face',cropped_face)
        resized_frame = cv2.resize(cropped_face,(width,height))
        resized_frame_gray = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)
        training_data.append([resized_frame_gray,isMainPerson(is_main_person)])
        
        cv2.imshow('resized_frame',resized_frame)

        if len(training_data)%100==0:
            print("Saved Data at {}".format(len(training_data)))
            np.save(file_name,training_data)


    
    if cv2.waitKey(10) and 0xFF==ord('q') :
        break

cap.release()
cv2.destroyAllWindows()
