# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:00:30 2020

@author: Aritz

LOADING THE MODEL, TESTING AND EVALUATING IT

This code can be run to load a model, predict labels and evaluate results
It also saves all predictions
Classification report added to analyze performance of each class

"""

import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import load_model
from load_dataset import load_images
from mask import *
import matplotlib.patches as mpatches
from models import weighted_categorical_crossentropy
import keras.losses

#############################################################################

model_name='name'
MODEL_PATH='C://Users/Aritz/Desktop/Project/models/{0}.h5'.format(model_name)
custom_loss = keras.losses.penalized_loss = weighted_categorical_crossentropy()
model = load_model(MODEL_PATH, custom_objects={'wcce': custom_loss})
print ('Model correctly loaded')

TRAIN_PATH = 'C://Users/Aritz/Desktop/Project/Images/Train/' #training images dataset path
TEST_PATH  = 'C://Users/Aritz/Desktop/Project/Images/Test/' #testing images dataset path
TEST_PREDICTIONS_PATH = 'C://Users/Aritz/Desktop/Project/Images/Outputs/' #testing outputs path
IMG_WIDTH = 256
IMG_HEIGHT = 256

images, test_images = load_images(TRAIN_PATH, TEST_PATH, TEST_PREDICTIONS_PATH, IMG_WIDTH, IMG_HEIGHT)
#create_labels() from mask.py. Used for representation, but not for the model training
#shape (ix, height, width, 3), like images 
print('Creating image labels...')
labels = create_labels(images)
print('Creating test image labels...')
test_labels = create_labels_noStat_noPrint(test_images)
print('Creating test image masks...')
test_masks = create_masks(test_images)

print('Testing on {0} images'.format(len(test_images))) 
test_outputs = model.predict(test_images, verbose=1)
#These predictions are masks
#output_to_label() from mask.py is used to convert them to representable labels
print('Creating predicted labels of test images...')
test_outputs_labels=output_to_label(test_outputs)
 
#Legend
red_patch = mpatches.Patch(color=[1, 0.2, 0.2], label='Cluster')
blue_patch = mpatches.Patch(color=[0,0.5,1.], label='Hot pixel')
green_patch = mpatches.Patch(color=[0.35,1.,0.25], label='Glowing')
black_patch = mpatches.Patch(color=[0./255, 0./255, 0./255], label='Background')

# Check prediction of a RANDOM TRAINING sample
ix = random.randint(0, len(images)-1)
image=images[ix][np.newaxis, ...]
train_outputs = model.predict(image, verbose=1)
print('Creating predicted label of training image {0}...'.format(ix+1))
train_outputs_labels=output_to_label(train_outputs)
fig, ax = plt.subplots(1, 3, figsize=(20, 10))
ax[0].imshow(images[ix])
ax[0].set_title('Training image {0}'.format(ix+1), fontsize=25);
ax[0].set_xlabel('pixels', fontsize=16)
ax[0].set_ylabel('pixels', fontsize=16)
ax[1].imshow(np.squeeze(labels[ix]))
ax[1].set_title('Training label {0}'.format(ix+1), fontsize=25);
ax[1].set_xlabel('pixels', fontsize=16)
ax[1].set_ylabel('pixels', fontsize=16)
ax[2].imshow(np.squeeze(train_outputs_labels))
ax[2].set_title('Training predicted label {0}'.format(ix+1), fontsize=25);
ax[2].set_xlabel('pixels', fontsize=16)
ax[2].set_ylabel('pixels', fontsize=16)
plt.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.15), fontsize=18,\
           handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4)
plt.savefig(TEST_PREDICTIONS_PATH+'Train_{0}'.format(ix+1))
plt.show()

# Checking and saving ALL TEST samples
for ix in range (len(test_outputs)):
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(test_images[ix])
    ax[0].set_title('Test image {0}'.format(ix+1), fontsize=25);
    ax[0].set_xlabel('pixels', fontsize=16)
    ax[0].set_ylabel('pixels', fontsize=16)
    ax[1].imshow(np.squeeze(test_labels[ix]))
    ax[1].set_title('Test label {0}'.format(ix+1), fontsize=25);
    ax[1].set_xlabel('pixels', fontsize=16)
    ax[1].set_ylabel('pixels', fontsize=16)
    ax[2].imshow(np.squeeze(test_outputs_labels[ix]))
    ax[2].set_title('Test predicted label {0}'.format(ix+1), fontsize=25);
    ax[2].set_xlabel('pixels', fontsize=16)
    ax[2].set_ylabel('pixels', fontsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.15), fontsize=18,\
           handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4)
    plt.savefig(TEST_PREDICTIONS_PATH+'Test_{0}'.format(ix+1))
    plt.show()

#Model evaluation   
evaluation = model.evaluate(test_images, test_masks, batch_size=16)
print('The accuracy of the model on the test set is: ',\
      evaluation[1]*100,'%')
print('The loss of the model on the test set is: ',\
      evaluation[0])
    
#Classification report
#Will help identifiying the misclassified classes in more detail. 
#Able to observe for which class the model performed better or worse.
number_to_class = ['background', 'glowing', 'hot pixel', 'cluster']
#convert 3D matrixes with values [0,1,2,3] to 1D arrays
test_max_masks=get_max_in_mask(test_masks)
test_max_outputs=get_max_in_mask(test_outputs)
test_masks_array=test_max_masks.ravel()
test_outputs_array=test_max_outputs.ravel()
from sklearn.metrics import classification_report
print(classification_report(y_true = test_masks_array, y_pred = test_outputs_array, target_names=number_to_class))            
#Recal:"how many of this class you find over the whole number of element of
# this class"
#Precision:"how many are correctly classified among that class"
#F1-score:the harmonic mean between precision & recall. Good on inbalanced sets, like this one
#Support:the number of occurence of the given class in your dataset
