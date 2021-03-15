import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

IMG_SIZE = 300

def load_data():
    DATADIR = './COVID-19 Radiography Database/train'
    CATEGORIES = ['covid', 'normal', 'pneumonia']
    
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

    #TEST DATA
    DATADIR = './COVID-19 Radiography Database/test'
    CATEGORIES = ['covid', 'normal', 'pneumonia']

    test_data = []
    shapes = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            shapes.append(img_array.shape)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            test_data.append([new_array, class_num])
   

    import random
    random.shuffle(training_data)
    random.shuffle(test_data)

    train_X = []
    train_y = []

    test_X = []
    test_y = []

    for features,label in training_data:
        train_X.append(features)
        train_y.append(label)

    for features,label in test_data:
        test_X.append(features)
        test_y.append(label)
    
    train_X = np.array(train_X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_X = np.array(test_X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return (train_X,train_y),(test_X,test_y)


if __name__ == '__main__':
    load_data()