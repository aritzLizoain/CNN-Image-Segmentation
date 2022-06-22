# -*- coding: utf-8 -*-
"""
//////////////////////////////////////////////////////////////////////////////////////////
// Original author: Aritz Lizoain
// Github: https://github.com/aritzLizoain
// My personal website: https://aritzlizoain.github.io/
// Description: CNN Image Segmentation
// Copyright 2020, Aritz Lizoain.
// License: MIT License
//////////////////////////////////////////////////////////////////////////////////////////

Working directory must be where all files are located
This code can be run to check both training and augmented labels (uncomment last section)
"""

import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
from mask import *

##############################################################################

def augmentation_sequence_Color(images, labels):
    labels = labels.astype(np.uint8)
    seq = iaa.Sequential([iaa.Dropout2d(p=0.70)])  #, iaa.Flipud(0.8),\
                          #iaa.OneOf([iaa.Rotate((270, 270))]) iaa.Fliplr(0.8),
                          #iaa.Rotate((90, 90)) only invert in order to avoid weight issues and masks
    return seq(images=images, segmentation_maps=labels)

#----------------------------------------------------------------------------

def augmentation_Color(images, labels, TEST_PREDICTIONS_PATH = ''):
    print("Applying data augmentation: dropout, rotation, flip.")
    images_aug, labels_aug = augmentation_sequence_Color(images=images, labels=labels)
    labels_aug = labels_aug.astype(np.float64)

    # Perform a sanity check on a random AUGMENTED sample
    # ix = random.randint(0, len(images_aug)-1)
    red_patch = mpatches.Patch(color=[1, 0.2, 0.2], label='Cluster')
    blue_patch = mpatches.Patch(color=[0,0.5,1.], label='Hot pixel')
    green_patch = mpatches.Patch(color=[0.35,1.,0.25], label='Glowing')
    black_patch = mpatches.Patch(color=[0./255, 0./255, 0./255], label='Background')
    # for ix in range(0,len(labels)):
    #     fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    #     ax[0,0].imshow(rgb2gray(images[ix]), cmap="gray") #rgb2gray & , cmap="gray" if grayscale
    #     ax[0,0].set_title('Training image: {0}'.format(ix+1), fontsize=18);
    #     ax[0,0].set_xlabel('pixels', fontsize=10)
    #     ax[0,0].set_ylabel('pixels', fontsize=10)
    #     ax[0,0].tick_params(axis='both', which='major', labelsize=10)
    #     ax[0,1].imshow(rgb2gray(images_aug[ix]), cmap="gray") #rgb2gray & , cmap="gray" if grayscale
    #     ax[0,1].set_title('Augmented image: {0}'.format(ix+1), fontsize=18);
    #     ax[0,1].set_xlabel('pixels', fontsize=10)
    #     ax[0,1].set_ylabel('pixels', fontsize=10)
    #     ax[0,1].tick_params(axis='both', which='major', labelsize=10)
    #     ax[1,0].imshow(images_aug[ix]) #rgb2gray & , cmap="gray" if grayscale
    #     ax[1,0].set_title('Augmented image: {0}'.format(ix+1), fontsize=18);
    #     ax[1,0].set_xlabel('pixels', fontsize=10)
    #     ax[1,0].set_ylabel('pixels', fontsize=10)
    #     ax[1,0].tick_params(axis='both', which='major', labelsize=10)
    #     # ax[1,0].imshow(labels[ix])
    #     # ax[1,0].set_title('Training label: {0}'.format(ix+1), fontsize=18);
    #     # ax[1,0].set_xlabel('pixels', fontsize=10)
    #     # ax[1,0].set_ylabel('pixels', fontsize=10)
    #     # ax[1,0].tick_params(axis='both', which='major', labelsize=10)
    #     ax[1,1].imshow(labels_aug[ix])
    #     ax[1,1].set_title('Augmented label: {0}'.format(ix+1), fontsize=18);
    #     ax[1,1].set_xlabel('pixels', fontsize=10)
    #     ax[1,1].set_ylabel('pixels', fontsize=10)
    #     ax[1,1].tick_params(axis='both', which='major', labelsize=10)
    #     plt.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.15), fontsize=18,\
    #                 handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4)
    #     plt.savefig(TEST_PREDICTIONS_PATH+'Augmentation')
    #     plt.show()  
    all_images = np.append(images , images_aug, axis=0 )
    all_labels= np.append(labels, labels_aug, axis=0)
    return all_images, all_labels

#----------------------------------------------------------------------------

def augmentation_sequence_Invert(images, labels):
    labels = labels.astype(np.uint8)
    seq = iaa.Sequential([iaa.Invert(p=1, per_channel=0.6)])  #, iaa.Flipud(0.8),\
                          #iaa.OneOf([iaa.Rotate((270, 270))]) iaa.Fliplr(0.8),
                          #iaa.Rotate((90, 90)) only invert in order to avoid weight issues and masks
    return seq(images=images, segmentation_maps=labels)

#----------------------------------------------------------------------------

def augmentation_Invert(images, labels, TEST_PREDICTIONS_PATH = ''):
    print("Applying data augmentation: invert, dropout, logContrast, hue, gammaContrast.")
    images_aug, labels_aug = augmentation_sequence_Invert(images=images, labels=labels)
    labels_aug = labels_aug.astype(np.float64)

        # Perform a sanity check on a random AUGMENTED sample
    ixn = random.randint(1, len(images_aug)-1)
    red_patch = mpatches.Patch(color=[1, 0.2, 0.2], label='Cluster')
    blue_patch = mpatches.Patch(color=[0,0.5,1.], label='Hot pixel')
    green_patch = mpatches.Patch(color=[0.35,1.,0.25], label='Glowing')
    black_patch = mpatches.Patch(color=[0./255, 0./255, 0./255], label='Background')
    for ix in range(ixn,ixn+1): #only one: ixn,ixn+1
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0,0].imshow(rgb2gray(images[ix]), cmap="gray") #rgb2gray & , cmap="gray" if grayscale
        ax[0,0].set_title('Training image: {0}'.format(ix+1), fontsize=18);
        ax[0,0].set_xlabel('pixels', fontsize=10)
        ax[0,0].set_ylabel('pixels', fontsize=10)
        ax[0,0].tick_params(axis='both', which='major', labelsize=10)
        ax[0,1].imshow(rgb2gray(images_aug[ix]), cmap="gray") #rgb2gray & , cmap="gray" if grayscale
        ax[0,1].set_title('Augmented image: {0}'.format(ix+1), fontsize=18);
        ax[0,1].set_xlabel('pixels', fontsize=10)
        ax[0,1].set_ylabel('pixels', fontsize=10)
        ax[0,1].tick_params(axis='both', which='major', labelsize=10)
        # ax[1,0].imshow(images_aug[ix]) #rgb2gray & , cmap="gray" if grayscale
        # ax[1,0].set_title('Augmented image: {0}'.format(ix+1), fontsize=18);
        # ax[1,0].set_xlabel('pixels', fontsize=10)
        # ax[1,0].set_ylabel('pixels', fontsize=10)
        # ax[1,0].tick_params(axis='both', which='major', labelsize=10)
        ax[1,0].imshow(labels[ix])
        ax[1,0].set_title('Training label: {0}'.format(ix+1), fontsize=18);
        ax[1,0].set_xlabel('pixels', fontsize=10)
        ax[1,0].set_ylabel('pixels', fontsize=10)
        ax[1,0].tick_params(axis='both', which='major', labelsize=10)
        ax[1,1].imshow(labels_aug[ix])
        ax[1,1].set_title('Augmented label: {0}'.format(ix+1), fontsize=18);
        ax[1,1].set_xlabel('pixels', fontsize=10)
        ax[1,1].set_ylabel('pixels', fontsize=10)
        ax[1,1].tick_params(axis='both', which='major', labelsize=10)
        plt.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.15), fontsize=18,\
                    handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4)
        plt.savefig(TEST_PREDICTIONS_PATH+'Augmentation')
        plt.show()  
    all_images = np.append(images , images_aug, axis=0 )
    all_labels= np.append(labels, labels_aug, axis=0)
    return all_images, all_labels

#----------------------------------------------------------------------------

#Same augmentation but without displaying anything on screen
def augmentation_noPrint(images, labels, TEST_PREDICTIONS_PATH = ''):
    images_aug, labels_aug = augmentation_sequence_Color(images=images, labels=labels)
    labels_aug = labels_aug.astype(np.float64)
    all_images = np.append(images , images_aug, axis=0 )
    all_labels= np.append(labels, labels_aug, axis=0)
    return all_images, all_labels
       

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
VISUALIZE AUGMENTATION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# import os
# import sys
# import numpy as np
# import cv2
# from skimage.transform import resize
# from mask import *

#############################################################################

# TRAIN_PATH = '' #training images dataset path
# TEST_PATH  = '' #testing images dataset path
# TEST_PREDICTIONS_PATH = '' #testing outputs path
# IMG_WIDTH = 256
# IMG_HEIGHT = 256

# images, test_images = load_images(TRAIN_PATH, TEST_PATH, TEST_PREDICTIONS_PATH, IMG_WIDTH, IMG_HEIGHT)
# labels = create_labels(images) #create_labels() from mask.py
# images_aug, labels_aug = augmentation(images=images, labels=labels)
