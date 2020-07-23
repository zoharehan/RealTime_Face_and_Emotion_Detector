import cv2
from random import randrange
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import sklearn
from tensorflow.keras import layers, Sequential, Model

#trained data loading from opencv's library
x_train = cv2.CascadeClassifier('/Users/zoharehan/PycharmProjects/face_detect/haarcascade_frontalface_default.xml')
em = "/Users/zoharehan/face_model.h5"
emotion_model = keras.models.load_model(em)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#test image to detect face
#test_img = cv2.imread('zoha.jpg')
#'0' gives us our default webcam so laptop's camera!
live_video = cv2.VideoCapture(0)

while True:
    frame_read_bool,frame = live_video.read()
    #the read function returns successfully read(boolean) and the current frame

    #black and white
    b_w_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #returns coordinates of multiple pictures regardless of scale
    #topleft, width and height
    face_detect = x_train.detectMultiScale(b_w_img)
    for (x, y, a, b) in face_detect:
        color = (randrange(256),randrange(256),randrange(256))
        cv2.rectangle(frame, (x,y),(x+a,y+b),color,6)
        #instead of a tuple we need an array with 4 elements
        img_fix = b_w_img[y:y + b, x:x + a]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(img_fix, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        #cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2, cv2.LINE_AA)
    cv2.imshow("Face and Emotion Detector", frame)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
live_video.release()
cv2.destroyAllWindows()