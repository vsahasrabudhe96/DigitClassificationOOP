import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import cv2

'''
Preprocessing the data, converting thae data in numpy arrays and also reshaping and resizing as required for a CNN
'''
class MNIST(object):
    data_root = '/home/varun/Downloads/MNIST/'

    def __init__(self, root=None):

        self.root = root        
        if not self.root:
            if os.path.exists(MNIST.data_root):
                self.root = MNIST.data_root
            else:
                print("Path {} does not exist".format(MNIST.data_root))
    def data_load(self):
        self.train_df = pd.read_csv(os.path.join(self.root,'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.root,'test.csv'))
    
    def head_tail(self,head=False,tail=False):
        if head:
            print('--------Top 5 rows of training data-------')
            print(self.train_df.head())
            print('--------Top 5 rows of test data--------')
            print(self.test_df.head())
        if tail:
            print('--------Bottom 5 rows of training data-------')
            print(self.train_df.tail())
            print('--------Bottom 5 rows of test data--------')
            print(self.test_df.tail())

    def split_X_y(self):
        self.X_train = self.train_df.drop('label',axis=1).values
        self.y_train = self.train_df['label']
    
    def prep_test(self):
        self.X_test = self.test_df.values
    
    def reshape(self,train=False,test=False):
        if train:
            num_images = self.X_train.shape[0] # examples/images
            m = self.X_train.shape[1] # pixels
            max_pixel = self.X_train.max().max() # maximum pixel value for rescaling
            img_dim = np.sqrt(m).astype(int) #image dimensions
            self.X_train = self.X_train.reshape((num_images,img_dim,img_dim,1)) / max_pixel
        if test:
            num_images = self.X_test.shape[0] # examples/images
            m = self.X_test.shape[1] # pixels
            max_pixel = self.X_test.max().max() # maximum pixel value for rescaling
            img_dim = np.sqrt(m).astype(int) #image dimensions
            self.X_test = self.X_test.reshape((num_images,img_dim,img_dim,1)) / max_pixel
        #return X
    
    def plot(self,i,train=False,test=False):
        if train:
            for i in range(i):
                plt.imshow((self.X_train[i][:,:,-1]))
                print('Label: {}'.format(self.y_train[i]))
                plt.show()
        if test:
            for i in range(i):
                plt.imshow((self.X_test[i][:,:,-1]))
                #print('Label: {}'.format(self.y_train[i]))
                plt.show()