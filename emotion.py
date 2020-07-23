import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import sklearn
from tensorflow.keras import layers, Sequential, Model
from sklearn.model_selection import train_test_split

from keras.models import Sequential 
from keras.layers import Conv2D #Convolution operation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense 
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score


#a function to use the "usage" column in the data to split it into training and testing
def data(data_path):
    a = pd.read_csv(data_path) 
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i in range(len(a["Usage"])):
        image_data = np.asarray([int(x) for x in a["pixels"][i].split()]).reshape(48, 48)
        image_data =image_data.astype(np.uint8)/255.0
        if (a["Usage"][i] == "PrivateTest"):
            test_data.append(image_data)
            test_labels.append(int(a["emotion"][i]))
        else:
            train_data.append(image_data)
            train_labels.append(int(a["emotion"][i]))

    test_data = np.expand_dims(test_data, -1)
    test_labels = to_categorical(test_labels, num_classes = 7)
    train_data = np.expand_dims(train_data, -1)   
    train_labels = to_categorical(train_labels, num_classes = 7)
    
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)
  
#loading the data
data_path = '/Users/zoharehan/Downloads/fer2013.csv'
training_data, training_labels, testing_data, testing_labels = data(data_path)


#setting up the convolutional neural network

learning_rate = 0.001


model = Sequential()
    
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.01)))
model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))
    
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
    
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
    
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
    
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
adam = optimizers.Adam(lr = learning_rate)
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
print(model.summary())

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=6, mode='auto')

model.fit(training_data,training_labels,epochs=25,batch_size=64,validation_split = 0.2,
          shuffle = True, callbacks=[lr_reducer, early_stopper])

predicted_test_labels = np.argmax(model.predict(testing_data), axis=1)
testing_labels = np.argmax(testing_labels, axis=1)
print ("Accuracy score = ", accuracy_score(testing_labels, predicted_test_labels))

model.save('face_model.h5')
