# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:00:30 2020

@author: Aritz Lizoain

ARCHITECTURES: UNet

"""
import numpy as np 
import os
import matplotlib.pyplot as plt   

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
import keras.losses
from keras import regularizers #fixing overfitting with L2 regularization
import keras.backend as K

#############################################################################

#Imbalanced dataset --> weighted loss function cross entropy is needed
#Images too biased towards the first class (background ~95%)

#WEIGHTED LOSS FUNCTION CROSS ENTROPY
#Taken from: https://stackoverflow.com/questions/61309991/how-to-use-weighted-categorical-crossentropy-loss-function
def weighted_categorical_crossentropy(weights= [1.,1.,1.,1.]):
    #print('The used loss function is: weighted categorical crossentropy')
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        if not K.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

#----------------------------------------------------------------------------

"""
UNET

It takes images (n_img, h, w, 3(rgb)) and masks (n_img, h, w, n_classes) for training
Output has shape (n_img, h, w, n_classes)

Comments are prepared to change number of layers

Modified network. Original from: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
"""

def unet(pretrained_weights = None, input_size = (256,256,3), weights= [1.,1.,1.,1.],\
         activation='relu', dropout=0, loss='categorical_crossentropy', optimizer='adam',\
             dilation_rate=(1,1), reg=0.01):
    
    inputs = Input(input_size)
    s = Lambda(lambda x: x / 255) (inputs) #this is the input of the first layer. Be sure to change it if layers are added/removed
    
    #CONTRACTIVE Path (ENCODER)    
    
    # cm3 = Conv2D(2, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (s)
    # cm3 = Dropout(dropout) (cm2)
    # cm3 = Conv2D(2, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (cm2)
    # pm3 = MaxPooling2D((2, 2)) (cm2)
    # pm3 = BatchNormalization()(pm2)
    
    # cm2 = Conv2D(4, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (s)
    # cm2 = Dropout(dropout) (cm2)
    # cm2 = Conv2D(4, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (cm2)
    # pm2 = MaxPooling2D((2, 2)) (cm2)
    # pm2 = BatchNormalization()(pm2)

    # cm1 = Conv2D(8, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (s)
    # cm1 = Dropout(dropout) (cm1)
    # cm1 = Conv2D(8, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (cm1)
    # pm1 = MaxPooling2D((2, 2)) (cm1)
    # pm1 = BatchNormalization()(pm1)
    
    # c0 = Conv2D(16, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (s)
    # c0 = Dropout(dropout) (c0)
    # c0 = Conv2D(16, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c0)
    # p0 = MaxPooling2D((2, 2)) (c0)
    # p0 = BatchNormalization()(p0)

    c1 = Conv2D(32, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (s)
    c1 = Dropout(dropout) (c1)
    c1 = Conv2D(32, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = BatchNormalization()(p1)

    c2 = Conv2D(64, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p1)
    c2 = Dropout(dropout) (c2)
    c2 = Conv2D(64, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = BatchNormalization()(p2)

    c3 = Conv2D(128, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p2)
    c3 = Dropout(dropout) (c3)
    c3 = Conv2D(128, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = BatchNormalization()(p3)

    c4 = Conv2D(256, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p3)
    c4 = Dropout(dropout) (c4)
    c4 = Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = BatchNormalization()(p4)

    c5 = Conv2D(512, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p4)
    c5 = Dropout(dropout) (c5)
    c5 = Conv2D(512, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c5)


    #EXPANSIVE Path (DECODER)
    
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u6)
    c6 = Dropout(dropout) (c6)
    c6 = Conv2D(256, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c6)
    
    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u7)
    c7 = Dropout(dropout) (c7)
    c7 = Conv2D(128, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c7)
    
    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u8)
    c8 = Dropout(dropout) (c8)
    c8 = Conv2D(64, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u9)
    c9 = Dropout(dropout) (c9)
    c9 = Conv2D(32, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c9)
    
    # u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c9)
    # u10 = concatenate([u10, c0], axis=3)
    # c10 = Conv2D(16, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u10)
    # c10 = Dropout(dropout) (c10)
    # c10 = Conv2D(16, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c10)
    
    # u11 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c10)
    # u11 = concatenate([u11, cm1], axis=3)
    # c11 = Conv2D(8, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u11)
    # c11 = Dropout(dropout) (c11)
    # c11 = Conv2D(8, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c11)

    # u12 = Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same') (c10)
    # u12 = concatenate([u12, cm2], axis=3)
    # c12 = Conv2D(4, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u12)
    # c12 = Dropout(dropout) (c12)
    # c12 = Conv2D(4, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c12)

    # u13 = Conv2DTranspose(2, (2, 2), strides=(2, 2), padding='same') (c11)
    # u13 = concatenate([u12, cm2], axis=3)
    # c13 = Conv2D(2, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u12)
    # c13 = Dropout(dropout) (c12)
    # c13 = Conv2D(2, (3, 3), activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c12)
 
    #softmax as activaition in the last layer
    outputs = Conv2D(4, (1, 1), activation='softmax') (c9) #this is the output of the last layer. Be sure to change it if layers are added/removed

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss=loss,\
                  metrics = ['accuracy'])
    #model.summary() 
    
    if(pretrained_weights):
        
        print('Using {0} pretrained weights'.format(pretrained_weights))
    
        model.load_weights(pretrained_weights)

    return model

