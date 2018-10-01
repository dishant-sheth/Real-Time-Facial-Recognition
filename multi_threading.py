# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

#CNN requirements
from keras.models import Model, Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation

#Image processing requirements
#from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

#Visulaization requirement
#import matplotlib.pyplot as plt

#File management
from os import listdir

import threading

label = ''
frame = None

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    
    def run_human_faces(self):
        #Picture database
        pictures = "/Users/dishantsheth/Desktop/Real Time Facial Recognition/Pictures/"
        
        humans = dict()
        for file in listdir(pictures):
            human, extension = file.split('.')
            humans[human] = self.model.predict(self.preprocess_image('/Users/dishantsheth/Desktop/Real Time Facial Recognition/Pictures/%s.jpg' % (human)))[0,:]
        
        self.humans = humans
        print("Human representations retrieved successfully")
    
    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
        
    def findCosineSimilarity(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
    def run(self):
        global label
        
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))
        
        model.load_weights('vgg_face_weights.h5')
        
        
        self.model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
        
        self.run_human_faces()
        
        while(~(frame is None)):
            label = self.predict(frame)
            
    def predict(self, frame):
        result = ''
        img_pixels = image.img_to_array(frame)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        captured_representation = self.model.predict(img_pixels)[0,:]
        found = 0
        for i in self.humans:
            human_name = i
            representation = self.humans[i]
            
            similarity = self.findCosineSimilarity(representation, captured_representation)
            
            if similarity < 0.3:
                result = human_name
                print(result)
                found = 1
                break
            
        if found == 0:
            result = 'No match found.'
            
        return result
    
    
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera status - 200")
else:
    cap.open()
    
color = (67,67,67)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


keras_thread = MyThread()
keras_thread.start()

while(True):
    ret, img = cap.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    for (x,y,w,h) in faces:
        if w > 130:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            detected_face = img[int(y):int(y+h), int(x):int(x+w)]
            detected_face = cv2.resize(detected_face, (224, 224))
            frame = detected_face
            
            cv2.putText(img, label, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if(cv2.waitKey(1) and 0xFF == ord('q')):
                break
    
    cv2.imshow('Multithreaded Realtime Face Recognition',img)
    if(cv2.waitKey(1) and 0xFF == ord('q')):
        break

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
