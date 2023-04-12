from flask import Flask, render_template, Response
import os
import cv2
import numpy

app = Flask(__name__)

camera = cv2.VideoCapture(0)


faceDet = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def gen_frames():
    # box_x, box_y, box_w, box_h = 200, 150, 200, 200
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:
            # convert to gscale
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # get face corners
            # 1.5 = scale factor, 5 = minimum neighbors
            faces = faceDet.detectMultiScale(imgGray, 1.3, 5)

            # draw box
            for(x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # cv2.imshow('app.py', frame)
            # cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: immage/jpeg\r\n\r\n' + frame + b'\r\n')


            
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(select_detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)