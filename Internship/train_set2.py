import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
from datetime import timedelta
# Get a reference to webcam #0
video_capture = cv2.VideoCapture(0)
file_name = "logs.npy"

file_name_encodings = "encodings.npy"
logs = np.array([])



if os.path.isfile(file_name):
    print("Loading Previously Saved File for Logs")
    logs = np.array(np.load(file_name))
else:
    print("No Previosly Saved File for Logs .. Creating New File")

if os.path.isfile(file_name_encodings):
    print("Loading Previously Saved File for Known Faces")
    known_face_encodings = list(np.array(np.load(file_name_encodings))[:,1])
    known_face_names = list(np.array(np.load(file_name_encodings)[:,0])
                            



# Load a sample picture and learn how to recognize it.
ajinkya_image = face_recognition.load_image_file("Ajinkya.jpg")
ajinkya_encodings = face_recognition.face_encodings(ajinkya_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    ajinkya_encodings
]
known_face_names = [
    "Ajinkya"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
save_length =2

def log_record(name,time):
    global logs
    logs = np.append(logs,[[name,time]])
    logs = logs.reshape((len(logs)//2,2))

    print("Appending logs..")
    if len(logs)%save_length==0:
        print("Saving Logs")
        np.save(file_name,logs)
    
def save_face(name,encodings):
                            print("Saving Face")
    


while True:
   
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            if len(logs)==0:
                log_record(name,datetime.now())
            elif name not in logs[:,0]:
                log_record(name,datetime.now())

            else:
                idx = list(logs[:,0]).index(name)
                t = logs[:,1][idx]
                if datetime.now()-t>timedelta(0,1800,0):
                    log_record(name,datetime.now())
                    


                
            face_names.append(name)

    process_this_frame = not process_this_frame


    
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
