from flask import Flask, render_template, Response
import cv2





# app = Flask(__name__)
camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 420)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


    # box_x, box_y, box_w, box_h = 200, 150, 200, 200
while True:
    success, frame = camera.read()
    
    # convert to gscale
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get face corners
    # 1.3 = scale factor, 5 = minimum neighbors
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)
    # print(faces)

    # # draw box
    for(x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('face_detect.py', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()