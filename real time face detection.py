# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 05:51:57 2020

@author: Manish Singh
"""
import cv2

face_cascades  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces  = face_cascades.detectMultiScale(gray_frame, 1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(255,255,0), 2)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    
    
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,0,255),2)
             
    cv2.imshow('img', frame)
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
camera.release()
cv2.destroyAllWindows()
