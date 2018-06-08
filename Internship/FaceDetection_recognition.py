import cv2
import numpy as np
import os
import face_recognition
cap = cv2.VideoCapture(0)
face_cascades = cv2.CascadeClassifier('face.xml')
file_name = "training_set.npy"


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
                resized_frame = cv2.resize(cropped_face,(128,128))
                encodings = face_recognition.face_encodings(resized_frame)[0]
                   
                resized_frame_gray = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)
                cv2.imshow('resized_frame',resized_frame)
                

	    
        if cv2.waitKey(10) and 0xFF==ord('q'):
               break

cap.release()
cv2.destroyAllWindows()

