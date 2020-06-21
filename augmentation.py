# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:33:00 2020

@author: Aritz Lizoain

AUGMENTATION

Working directory must be where all files are located.

Geometric augmentation: flip, crop, pad, scale, translate and rotate.

This code can be run to check both training and augmented labels (uncomment last section)

"""

import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches

##############################################################################

def augmentation_sequence(images, labels):
    print("Applying geometric data augmentation: flip, crop, pad, scale, translate, rotate.")
    labels = labels.astype(np.uint8)
    seq = iaa.OneOf([
        iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.2)]),
        iaa.CropAndPad(percent=(-0.05, 0.1),
                       pad_mode='constant',
                       pad_cval=(0, 0)),
        iaa.Crop(percent=(0, 0.1)),
        iaa.Sequential([
            iaa.Affine(
                     # scale images to 90-110% of their size,
                     # individually per axis
                     scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                     # translate by -10 to +10 percent (per axis)
                     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                     rotate=(-30, 30)  # rotate by -30 to +30 degrees
                     #shear=(-16, 16),  # shear by -16 to +16 degrees
                     # use nearest neighbour or bilinear interpolation (fast)
                     #order=[0, 1],
                     # if mode is constant, use a cval between 0 and 255 (0 for black)
                     #mode='constant',
                     #cval=(0, 0),
                     # use any of scikit-image's warping modes
                     # (see 2nd image from the top for examples)
                     ),
            iaa.Sometimes(0.3, iaa.Crop(percent=(0, 0.01)))])
        ]) 
    return seq(images=images, segmentation_maps=labels)

#----------------------------------------------------------------------------

def augmentation(images, labels, TEST_PREDICTIONS_PATH = 'C://Path/'):
    images_aug, labels_aug = augmentation_sequence(images=images, labels=labels)
    labels_aug = labels_aug.astype(np.float64)

    # Perform a sanity check on a random AUGMENTED sample
    ix = random.randint(0, len(images_aug)-1)
    red_patch = mpatches.Patch(color=[1, 0.2, 0.2], label='Cluster')
    blue_patch = mpatches.Patch(color=[0,0.5,1.], label='Hot pixel')
    green_patch = mpatches.Patch(color=[0.35,1.,0.25], label='Glowing')
    black_patch = mpatches.Patch(color=[0./255, 0./255, 0./255], label='Background')
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0,0].imshow(images[ix])
    ax[0,0].set_title('Training image: {0}'.format(ix+1), fontsize=18);
    ax[0,0].set_xlabel('pixels', fontsize=10)
    ax[0,0].set_ylabel('pixels', fontsize=10)
    ax[0,0].tick_params(axis='both', which='major', labelsize=10)
    ax[0,1].imshow(images_aug[ix])
    ax[0,1].set_title('Augmented image: {0}'.format(ix+1), fontsize=18);
    ax[0,1].set_xlabel('pixels', fontsize=10)
    ax[0,1].set_ylabel('pixels', fontsize=10)
    ax[0,1].tick_params(axis='both', which='major', labelsize=10)
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

# TRAIN_PATH = 'C://Path/' #training images dataset path
# TEST_PATH  = 'C://Path/' #testing images dataset path
# TEST_PREDICTIONS_PATH = 'C://Path/' #testing outputs path
# IMG_WIDTH = 256
# IMG_HEIGHT = 256

# images, test_images = load_images(TRAIN_PATH, TEST_PATH, TEST_PREDICTIONS_PATH, IMG_WIDTH, IMG_HEIGHT)
# labels = create_labels(images) #create_labels() from mask.py
# images_aug, labels_aug = augmentation(images=images, labels=labels)
