from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

#importing classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

#impoting drowsy model and emomodel
drowsy_model = load_model('Drowsy_model.h5')
emotion_model =load_model('E:/main project parvathy/common/common/model.h5')

#labels of both emo and drow
lbl=['Close', 'Open']
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height,width = frame.shape[:2] 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,minNeighbors = 3,scaleFactor = 1.1,minSize=(25,25))
    eyes = eye_cascade.detectMultiScale(gray,minNeighbors = 1,scaleFactor = 1.1)

    #emotion based deployment so we use faces as dataset contains face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        #captureing the face and resizing based on the modeled size 48*48
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction_emo = emotion_model.predict(roi)[0]
            label=emotion_labels[prediction_emo.argmax()]
            label_position = (x,y-10)
            cv2.putText(frame,label,label_position,font,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),font,1,(0,255,0),2)
    
    # detects drowsyness from extracting the features of eyes
    for (x,y,w,h) in eyes:
        eye = frame[y:y+h,x:x+w]
        eye = cv2.resize(eye,(80,80)) #we resize the extracted image to `80*80
        eye = eye/255
        eye = eye.reshape(80,80,3)
        eye = np.expand_dims(eye,axis=0)
        prediction = drowsy_model.predict(eye)
        if prediction[0][0]<=0.10:
                cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

            #Condition for Open
        elif prediction[0][1] > 0.30:
                cv2.putText(frame,"Closed",(30,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        

    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()












