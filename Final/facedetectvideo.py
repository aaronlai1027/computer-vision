import cv2 as cv
import sys

face_cascade = cv.CascadeClassifier('D:\Program Files\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('D:\Program Files\opencv\sources\data\haarcascades\haarcascade_eye.xml')

video_capture = cv.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.7, 5)
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
