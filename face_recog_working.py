import face_recognition
import cv2
import face_recognition as face_recognition
import numpy as np
import os
from datetime import datetime,timedelta
import random
import string
import time
# Get a reference to webcam #0
video_capture = cv2.VideoCapture(0)
file_name = "logs.npy"

file_name_encodings = "encodings.npy"
file_name_names = "names.npy"
logs = np.array([])


video_capture.set(cv2.CAP_PROP_FPS,120)


if os.path.isfile(file_name):
    print("Loading Previously Saved File for Logs")
    logs = np.array(np.load(file_name))
else:
    print("No Previosly Saved File for Logs .. Creating New File")
known_face_encodings=[]
known_face_names=[]
if os.path.isfile(file_name_encodings):
    print("Loading Previously Saved File for Known Faces")
    known_face_encodings = list(np.load(file_name_encodings))
    known_face_names = list(np.load(file_name_names))
else:
    print("No Previosly Saved File for Known Faces .. Creating New File")



# Load a sample picture and learn how to recognize it.
ajinkya_image = face_recognition.load_image_file("Ajinkya.jpg")
ajinkya_encodings = face_recognition.face_encodings(ajinkya_image)[0]

# Create arrays of known face encodings and their names


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
save_length =2


def save_faces():
    global known_face_encodings,known_face_names
    np.save(file_name_names,known_face_names)
    np.save(file_name_encodings,known_face_encodings)
    print("Saved Known Faces")



def log_record(name,time):
    global logs
    logs = np.append(logs,[[name,time]])
    logs = logs.reshape((len(logs)//2,2))

    print("Appending logs..")
    if len(logs)%save_length==0:
        print("Saving Logs")
        np.save(file_name,logs)
        save_faces()


    

    
def RandomIdGen():
    N = 10
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

delayedExecution=0
while True:
    t1 = time.time()
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        t2 = time.time()
        delayedExecution+=1
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if delayedExecution==60:
            face_encodings =face_recognition.face_encodings(rgb_small_frame, face_locations)
            delayedExecution=0
        print("Face matching took:{}".format(time.time()-t2))
        face_names = []
        
        for face_encoding in face_encodings:
           
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
                 
            

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            if len(logs)==0:
                log_record(name,datetime.now())
            elif name == "Unknown":
                name = RandomIdGen()
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
                log_record(name,datetime.now())
                
                

            else:
                idx = list(logs[:,0])[::-1].index(name)
                t = list(logs[:,1])[::-1][idx]
                if datetime.now()-t>timedelta(0,600,0):
                    log_record(name,datetime.now())
            face_names.append(name)      
        
    
                
            
    
    


    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    
    cv2.imshow('Video', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
