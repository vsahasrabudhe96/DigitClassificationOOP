'''
Making the machine learning model
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.activations import relu, softmax, sigmoid
import os
class Model(object):
    data_root = '/home/varun/Downloads/MNIST/'

    def __init__(self, root=None):

        self.root = root        
        if not self.root:
            if os.path.exists(Model.data_root):
                self.root = Model.data_root
            else:
                print("Path {} does not exist".format(Model.data_root))

    def build_model(self,INPUT_SHAPE,OUTDIM):
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=INPUT_SHAPE,activation='relu',filters=32,kernel_size=5,strides=1,padding='valid'))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
        self.model.add(Conv2D(filters=64,kernel_size=5,activation='relu',padding='valid'))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
        self.model.add(Conv2D(filters=128,kernel_size=3,activation='relu',padding='valid'))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling2D(pool_size=(1,1),strides=2,padding='valid'))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Dense(OUTDIM,activation='softmax'))
        self.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        print(self.model.summary())
        

    def train_model(self,EPOCHS,BATCH_SIZE,X,y,VAL_SPLIT):
        self.history = self.model.fit(
            X,y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VAL_SPLIT
        )
        return self.history, self.model

    def predict(self,model,X_test):
        self.y_pred = model.predict(X_test)
        return self.y_pred