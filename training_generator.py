# -*- coding: utf-8 -*-
"""
Created on Sun Dec 05 19:32:45 2020

@author: david
"""

from tensorflow.keras.utils import Sequence
import math
import cv2
import numpy as np


class TrainingGenerator(Sequence):

    def _standardize_images(self,images):
        return (images - np.mean(images,axis=(1,2),keepdims = 1)) / np.std(images,axis=(1,2),keepdims = 1)
    
    #images are either saved as an array of pixels, or an array of paths
    def __init__(self, augmentor_pipeline,images,labels,batch_size,img_size,normalize,data_aug):
        #print('Initializing generator')
        self.augmentor_pipeline = augmentor_pipeline
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.normalize = normalize
        self.data_aug = data_aug

    def __len__(self):
        return math.ceil(len(self.images) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) *self.batch_size]
        X = []
        for image in batch_images:
            #If images are saved as paths
            if len(image.shape) == 0:
                img = cv2.imread(image)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img,(self.img_size,self.img_size))
            #Images are saved as pixels array
            elif len(image.shape) >=2:
                #img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                img = cv2.resize(image,(self.img_size,self.img_size))
            
            w = img.shape[0]
            h = img.shape[1]
            
            img = np.reshape(img,(w,h))
            if self.data_aug:
                img = self.augmentor_pipeline._execute_with_array(img)
            img = np.reshape(img,(w,h,1))
            X.append(img)
        
        
        X = np.asarray(X)
        Y = np.asarray(batch_labels)
        if self.normalize:
                X = self._standardize_images(X)
        return X,Y