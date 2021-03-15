# -*- coding: utf-8 -*-
"""
Created on Sun Dec 05 18:58:08 2020

@author: david
"""

from data_process import load_data
import Augmentor
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2,EfficientNetB3,EfficientNetB7
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import  Input, Conv2D, MaxPooling2D,GlobalMaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Activation, MaxPool2D, AvgPool2D, Dropout,BatchNormalization
from training_generator import TrainingGenerator
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from data_process import IMG_SIZE

#CONSTANTS
BATCH_SIZE = 32 #Should be changed depending on hardware, decrease BATCH_SIZE if your GPU has not a lot of VRAM
EPOCHS = 100
LEARNING_RATE =1e-3
CHECKPOINT_DIRECTORY = 'weights'
TENSORBOARD_DIRECTORY = 'tensorboard'
HISTORY_DIRECTORY = 'history'
MODEL_DIRECTORY = 'model'
CONFUSION_MATRIX_DIRECTIORY = 'cmatrix'
METRICS_DIRECTORY = 'metrics'
USE_DATA_AUGMENTATION = False


#Returns the model, either naive or efficientnet
def getModel(modelName,img_size):
    def getNaiveModel():
        visible = Input(shape=(img_size, img_size, 1), dtype=tf.float32)
        conv = Conv2D(32, (5, 5))(visible)
        conv_act = Activation('relu')(conv)
        conv_maxpool = MaxPooling2D()(conv_act)
    
        conv = Conv2D(64, (3, 3))(conv_maxpool)
        conv_act = Activation('relu')(conv)
        conv_maxpool = MaxPooling2D()(conv_act)
    
        conv = Conv2D(64, (3, 3))(conv_maxpool)
        conv_act = Activation('relu')(conv)
        conv_maxpool = MaxPooling2D()(conv_act)
    
        conv = Conv2D(64, (3, 3))(conv_maxpool)
        conv_act = Activation('relu')(conv)
        conv_maxpool = MaxPooling2D()(conv_act)
    
        conv = Conv2D(64, (3, 3))(conv_maxpool)
        conv_act = Activation('relu')(conv)
        conv_maxpool = MaxPooling2D()(conv_act)
        
        gap2d = GlobalAveragePooling2D()(conv_maxpool)
        act = Activation('relu')(gap2d)
    
        fc1 = Dense(256)(act)
        act = Activation('relu')(fc1)
    
        # and a logistic layer
        predictions = Dense(3, activation='softmax',name='prediction')(act)
        
        # Create model.
        model = tf.keras.Model(visible, predictions, name=modelName)
        return model
    
    def getEfficientNetModelB2():
        model = EfficientNetB2(include_top=True,
                               weights=None,
                               classes=3, 
                               classifier_activation='softmax')
        return model
    def getEfficientNetModelB3():
        model = EfficientNetB3(include_top=True,
                               weights=None,
                               classes=3, 
                               classifier_activation='softmax')
        return model
    
    def getEfficientNetModelB7():
        model = EfficientNetB7(include_top=True,
                               weights=None,
                               classes=3, 
                               classifier_activation='softmax')
        return model
    if modelName == 'naive':
        return getNaiveModel(),modelName
    elif modelName == 'efficientNetB2':
        return getEfficientNetModelB2(),modelName
    elif modelName == 'efficientNetB3':
        return getEfficientNetModelB3(),modelName
    elif modelName == 'efficientNetB7':
        return getEfficientNetModelB7(),modelName
    else:
        raise Exception("Wrong model name! Either use 'naive' or 'efficientNet' as model name!")

#Data Augmentation operations
def getAugmentorPipeline():
        p = Augmentor.Pipeline()
        #stack operations...
        p.zoom(probability=0.25,min_factor=0.75,max_factor=1.25)
        p.random_brightness(probability=0.25,min_factor=0.75,max_factor=1.25)
        p.flip_left_right(probability=0.25)
        p.rotate(probability=0.25,max_left_rotation=10,max_right_rotation=10)
        p.shear(probability=0.25, max_shear_left=10, max_shear_right=10)
        p.random_erasing(probability=0.25,rectangle_area=0.25)
        return p
    
#Used for the learning rate schedular
def schedular(epoch):
    #for the first quarter of epochs, use the initial learning rate
    if epoch < EPOCHS * 0.25:
        return LEARNING_RATE
    #after the first quarter of epochs and before half of the epochs, multiply the learning rate by 0.2
    elif epoch < EPOCHS *0.5:
        return LEARNING_RATE*0.2
    #third quarter
    elif epoch < EPOCHS * 0.75:
        return LEARNING_RATE*0.04
    #last quarter
    return LEARNING_RATE*0.008

def splitData(X,Y,validation_ratio=0.25):
    data_size = len(X)
    indices = np.arange(data_size)
    #shuffle in place
    np.random.shuffle(indices)
    #shuffle input and output
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    train_size = int(data_size * (1-validation_ratio))
    
    #slice the shuffled input
    X_train,X_test = X_shuffled[:train_size],X_shuffled[train_size:]
    #slice the shuffled output
    Y_train,Y_test = Y_shuffled[:train_size],Y_shuffled[train_size:]
    
    assert len(X_train) + len(X_test) == data_size
    assert len(X_train) == len(Y_train)
    assert len(X_test) == len(Y_test)
    
    return X_train,X_test,Y_train,Y_test


def main():
    #X_train contains path to image
    (X_train,Y_train), (X_test,Y_test) = load_data()
    Y_train = to_categorical(Y_train,3)
    Y_test = to_categorical(Y_test,3)
    
    model,modelName = getModel('naive',IMG_SIZE)
    #print out info about the models(layer structures etc)
    model.summary()
    #Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data
    #Most commonly used, can be changed out by SGD or anything similar
    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    #best to use categorial loss for classification problems
    loss_fuction = 'categorical_crossentropy'
    loss=[loss_fuction]
    
    #Metrics to measure during training, we are only interested in the prediction accuracy for now
    metrics = {'prediction':'accuracy'}
    #set the optimizer, loss function, and the metrics to print when training
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics= metrics)
    
    #List of Utilities called at certain points during model training
    callbacks = [LearningRateScheduler(schedular), #schedule the learning rate
                 ModelCheckpoint(os.path.join(CHECKPOINT_DIRECTORY,modelName + '{epoch:02d}-{val_loss:.2f}.hdf5'),
                                 monitor='val_loss', #monitor validation loss
                                 verbose=1, #print fancy progressbar
                                 save_best_only=True, #self explanatory
                                 mode='auto', #the decision to overwrite current save file
                                 save_weights_only=True, #save only the weights, not full model
                                 save_freq = 'epoch' ), #save after every epoch
                 TensorBoard(log_dir=TENSORBOARD_DIRECTORY,
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True)]
                 
        
    p = getAugmentorPipeline()
    #stack operations...
    p.random_brightness(probability=0.25,min_factor=0.75,max_factor=1.25)
    p.flip_left_right(probability=0.25)
    p.rotate(probability=0.25,max_left_rotation=10,max_right_rotation=10)
    p.shear(probability=0.25, max_shear_left=10, max_shear_right=10)
    #p.zoom(probability=0.25, min_factor=1.1, max_factor=1.5)
    p.random_erasing(probability=0.25,rectangle_area=0.25)


    training_generator = TrainingGenerator(augmentor_pipeline=p,
                                                images=X_train,
                                                labels=Y_train,
                                                batch_size = BATCH_SIZE,
                                                img_size=IMG_SIZE,
                                                normalize=True,
                                                data_aug=True if USE_DATA_AUGMENTATION else False)
         
    validation_generator = TrainingGenerator(augmentor_pipeline=p,
                                                images=X_test,
                                                labels=Y_test,
                                                batch_size=BATCH_SIZE,
                                                img_size=IMG_SIZE,
                                                normalize=True,
                                                data_aug=False)
    print('Training model...')
    history = model.fit_generator(generator=training_generator,
                            steps_per_epoch = len(X_train) // BATCH_SIZE,
                            validation_data = validation_generator,
                            validation_steps =len(X_test) // BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            callbacks=callbacks,
                            workers=6,
                            use_multiprocessing = False,
                            shuffle = True,
                            initial_epoch=0,
                            max_queue_size =6
                            )
    #Confusion Matrix and Classification Report
    Y_pred = model.predict(x=validation_generator, steps=len(X_test) // BATCH_SIZE+1)
    y_pred = np.argmax(Y_pred, axis=1)
    y_test = np.argmax(Y_test, axis=1)
    print('Saving confusion matrix as {}.png'.format(modelName))
    #covid 0, normal 1, pneumonia 2
    target_names = ['normal','corona','pnemonia']
    
    c_matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(c_matrix, index = [i for i in target_names],
                  columns = [i for i in target_names])
    plt.figure(figsize = (10,10))
    plot = sns.heatmap(df_cm, annot=True)
    plot.figure.savefig(os.path.join(CONFUSION_MATRIX_DIRECTIORY,'{}.png'.format(modelName)))
    
    print(f'Saving classification report as {modelName}.txt')
    with open(os.path.join(METRICS_DIRECTORY,'{}.txt'.format(modelName)),'w') as f:
        f.write(classification_report(y_test, y_pred, target_names=target_names))

    
    
    print('Saving history and model...')
    #Save the history 
    with open(os.path.join(HISTORY_DIRECTORY,modelName + '.h5'),'wb') as f:
        pickle.dump(history.history,f)
    
    #Save the whole model
    model.save(os.path.join(MODEL_DIRECTORY,modelName + '.h5'))

if __name__ == "__main__":
    main()