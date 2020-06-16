# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:00:30 2020

@author: Aritz Lizoain

TRAINING

This is the MAIN CODE to train the model.

Working directory must be where all files are located: 
C://Users/Aritz/Desktop/Project

PROCESS:
    *Loads the images
    *Creates the labels (for representation) and masks (for training) from them
    *Applies augmentation on both images and labels
    *Trains the model with the defined hyperparameters.
    It takes images (n_img, h, w, 3(rgb)) and masks (n_img, h, w,n_classes)
    *Plots and saves the accuracy and loss over the training
    *Predicts on train and test images. Predictions shape (n_img, h, w, n_classes)
    *Converts predicted masks to labels with shape (n_img, h, w, 3(rgb))
    *Plots and saves original images, labels, and predicted label comparisons
    *Evaluates the model
    *Classification report added to analyze performance of each class

"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
DATA LOADING AND CHECKING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import random
import matplotlib.pyplot as plt
from mask import create_labels_noStat_noPrint, create_masks, create_labels
from load_dataset import load_images
from augmentation import augmentation

##############################################################################

"""
GENERATE THE DATA
"""
TRAIN_PATH = 'C://Users/Aritz/Desktop/Project/Images/Train/' #training images dataset path
TEST_PATH  = 'C://Users/Aritz/Desktop/Project/Images/Test/' #testing images dataset path
TEST_PREDICTIONS_PATH = 'C://Users/Aritz/Desktop/Project/Images/Outputs/' #testing outputs path
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS=4 #number of classes

#load_images() from load_dataset.py
images, test_images = load_images(TRAIN_PATH, TEST_PATH, TEST_PREDICTIONS_PATH, IMG_WIDTH, IMG_HEIGHT)

#create_labels() from mask.py. Used for representation, but not for the model training
#shape (ix, height, width, 3), like images 
print('Creating training image labels...')
labels = create_labels(images)
print('Creating test image labels...')
test_labels = create_labels_noStat_noPrint(test_images)
#augmentation() from augmentation.py
#Both image and label augmentation is done at the same time
#Augmented images and labels are added to the original datasets (but not saved in disk)
#images, labels = augmentation(images, labels)

#create_masks() from mask.py
#shape (ix, height, width, n_classes)
print('Creating image masks...')
masks = create_masks(images)
print('Creating test image masks...')
test_masks = create_masks(test_images)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
MODEL TRAINING

It takes images (n_img, h, w, 3(rgb)) and masks (n_img, h, w, n_classes) for training
Output has shape (n_img, h, w, n_classes)
Then it is converted and represented as a label (n_img, h, w, 3(rgb))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from models import unet, weighted_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from load_dataset import get_weights

##############################################################################

#HYPERPARAMETERS

#-Model-----------------------------------------------------------------------
split=0.2 # Validation and training dataset split (default 0.0 --> 0%)
pretrained_weights = None # (default None)
input_size = (IMG_HEIGHT, IMG_WIDTH, 3) # (default (256,256,3))
#get_weights() calculates weights with class frequencies in the dataset
#it takes the frequencies from the training set
weights = get_weights(images,test_images) # (default [1.0, 1.0, 1.0, 1.0])
activation = 'elu' # (default 'relu') Activation on last layer is always softmax
dropout = 0.20 # (default 0.0)
dilation_rate = (1,1) # (default (1,1))
loss = weighted_categorical_crossentropy(weights) # (default 'categorical crossentropy')
reg = 0.01 # (default 0.01) L2 regularization
# learning_rate =  0.0001 
optimizer = 'adadelta' # (default 'adam')

#-Fit-----------------------------------------------------------------------
epochs = 25
batch_size = 16
callbacks=[EarlyStopping(patience=5,verbose=1),\
           ReduceLROnPlateau(factor=0.1, patience=2, min_delta=0.01, min_lr=0.00001,verbose=1),\
               ModelCheckpoint('models/name.h5',verbose=1, save_best_only=True,\
                               save_weights_only=False),\
                   CSVLogger('models/name.log')] #DON'T OVERWRITE
    
#-----------------------------------------------------------------------------

model = unet(pretrained_weights = pretrained_weights, input_size = input_size, weights = weights,\
         activation=activation, dropout=dropout, loss=loss, optimizer=optimizer, dilation_rate=dilation_rate)

results = model.fit(images, masks, validation_split=split, epochs=epochs, batch_size=batch_size,\
                      callbacks=callbacks, shuffle=True) 
    
print('Model correctly trained and saved')    
    
plt.figure(figsize=(8, 8))
plt.title("Learning curve LOSS", fontsize=25)
plt.plot(results.history["loss"], label="Loss")
plt.plot(results.history["val_loss"], label="Validation loss")
p=np.argmin(results.history["val_loss"])
plt.plot( p, results.history["val_loss"][p], marker="x", color="r", label="best model")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend();
plt.savefig(TEST_PREDICTIONS_PATH+'Loss')

plt.figure(figsize=(8, 8))
plt.title("Learning curve ACCURACY", fontsize=25)
plt.plot(results.history["accuracy"], label="Accuracy")
plt.plot(results.history["val_accuracy"], label="Validation Accuracy")
plt.plot( p, results.history["val_accuracy"][p], marker="x", color="r", label="best model")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.legend();
plt.savefig(TEST_PREDICTIONS_PATH+'Accuracy')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL (same in load.py)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from mask import output_to_label, get_max_in_mask
import matplotlib.patches as mpatches

##############################################################################

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
ix_random_training = random.randint(0, len(images)-1)
image=images[ix_random_training][np.newaxis, ...]
train_outputs = model.predict(image, verbose=1)
print('Creating predicted label of training image {0}...'.format(ix_random_training+1))
train_outputs_labels=output_to_label(train_outputs)
fig, ax = plt.subplots(1, 3, figsize=(20, 10))
ax[0].imshow(images[ix_random_training])
ax[0].set_title('Training image {0}'.format(ix_random_training+1), fontsize=25);
ax[0].set_xlabel('pixels', fontsize=16)
ax[0].set_ylabel('pixels', fontsize=16)
ax[1].imshow(np.squeeze(labels[ix_random_training]))
ax[1].set_title('Training label {0}'.format(ix_random_training+1), fontsize=25);
ax[1].set_xlabel('pixels', fontsize=16)
ax[1].set_ylabel('pixels', fontsize=16)
ax[2].imshow(np.squeeze(train_outputs_labels))
ax[2].set_title('Training predicted label {0}'.format(ix_random_training+1), fontsize=25);
ax[2].set_xlabel('pixels', fontsize=16)
ax[2].set_ylabel('pixels', fontsize=16)
plt.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.15), fontsize=18,\
           handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4)
plt.savefig(TEST_PREDICTIONS_PATH+'Train_{0}'.format(ix_random_training+1))
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
