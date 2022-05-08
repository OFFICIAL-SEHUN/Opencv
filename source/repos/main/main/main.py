import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray,frame):
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE)

    for(x,y,w,h) in faces:
        cv2,cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        face_gray = gray[y:y+h,x:x+w]
        face_color=frame[y:y+h,x:x+w]

        eyes = eyeCascade.detectMultiScale(face_gray,1.1,3)

        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame

video_capture = cv2.VideoCapture(0)

while True:

    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    canvas = detect(gray,frame)

    cv2.imshow("haha",canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()