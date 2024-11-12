from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('yolov8n')
classNames = ['person']

while True:
    ret, img= cap.read()
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break


