import cv2
import numpy as np
import os
import face_recognition
from datetime import timedelat
cap = cv2.VideoCapture(0)
face_cascades = cv2.CascadeClassifier('face.xml')
file_name = "training_set.npy"

if os.path.isfile(file_name):
    print("Loading Previously Saved File")
    training_data = list(np.load(file_name))
else:
    print("No Previosly Saved File .. Creating New File")
    training_data = []

id = 0

        
        
while True:
        ret,frame = cap.read()
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	    
        faces = face_cascades.detectMultiScale(gray,1.3,5)

	
        if len(faces)!=0:
                for (x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        cropped_face = frame[y:y+h,x:x+w]
                cv2.imshow('frame',frame)
                
                resized_frame = cv2.resize(cropped_face,(128,128))
                cv2.imshow('Cropped Frame',resized_frame)
                unk_encodings = face_recognition.face_encodings(resized_frame)
                if unk_encodings != []:
                    print(unk_encodings)

	    
        if cv2.waitKey(10) and 0xFF==ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

