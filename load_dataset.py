# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:29:09 2020

@author: Aritz

LOAD TRAINING AND TESTING DATASET

get_weights calculates the weights for the loss function. Each class weight is
obtained as the inverse of its percentage over all the training samples.
Then the weights are normalized to the number of classes

"""

import os
import sys
import numpy as np
import cv2
from skimage.transform import resize

#############################################################################

def load_images(TRAIN_PATH='C://Users/Aritz/Desktop/Project/Images/Train/',\
                TEST_PATH='C://Users/Aritz/Desktop/Project/Images/Test/',\
                TEST_PREDICTIONS_PATH='C://Users/Aritz/Desktop/Project/Images/TS_output/',\
                IMG_WIDTH = 256, IMG_HEIGHT = 256):

    train_ids = next(os.walk(TRAIN_PATH))[2]
    test_ids = next(os.walk(TEST_PATH))[2]

    # Get and resize train images and masks
    images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,3), dtype=np.uint8)
    test_images = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    sys.stdout.flush()
    
    # # train images 
    for n,id_ in enumerate(train_ids):
        img = cv2.imread(TRAIN_PATH + id_)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        images[n] = img
        
        # # test images 
        for n,id_ in enumerate(test_ids):
            mask_ = cv2.imread(TEST_PATH + id_)
            mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True, mode='constant')
            test_images[n] = mask_  
    print('Dataset correctly loaded')
    return images, test_images

#------------------------------------------------------------------------------

def get_weights(images,test_images):
    from mask import get_percentages
    #all_images = np.concatenate((images, test_images)) to take both training and test images
    all_images=images #to take only training images
    unique_elements, percentage = get_percentages(all_images)
    inverse_percentages=1/percentage #the weights are inversely proportional to their frequency
    weights = inverse_percentages/sum(inverse_percentages)*len(unique_elements) #normalize to the number of classes
    return weights
