# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:00:30 2020

@author: Aritz Lizoain

Working directory must be where all files are located

PROCESS:
    *Loads the images
    *Creates the labels
    *(Optional) Applies augmentation on both images and labels
    *Trains the model with the defined hyperparameters.
    It takes images (n_img, h, w, 3(rgb)) and labels (n_img, h, w, n_classes)
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
from mask import *
from load_dataset import load_images #(unused in version 2.0)
from augmentation import *

##############################################################################

"""
GENERATE THE DATA
"""
TRAIN_PATH = '' #training images dataset path
TEST_PATH  = '' #testing images dataset path
TEST_PREDICTIONS_PATH = '' #testing outputs path
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS=4 #number of classes

"""
Load original dataset PNG IMAGES (used in version 1.0)
"""
# #load_images() from load_dataset.py
# images_original, test_images_original = load_images(TRAIN_PATH, TEST_PATH, TEST_PREDICTIONS_PATH, IMG_WIDTH, IMG_HEIGHT)

"""
Load original dataset ARRAYS (real pixel values, version 2.0)
"""
images_original = np.load('Images/Train/training_data.npy') 
test_images_original = np.load('Images/Test/test_data.npy') 
print('Dataset arrays correctly loaded')

"""
Create masks
"""
#create_masks() from mask.py
#shape (ix, height, width, n_classes)
print('Creating training image masks...') 
masks = create_masks(images_original)
print('Creating test image masks...')
test_masks = create_masks(test_images_original)

"""
Create labels
"""
#create_labels() from mask.py. Used for representation, but not for the model training
#shape (ix, height, width, 3), like images 
print('Creating training image labels...')
labels = create_labels(images_original)
# # To check the labels
# for ix in range(0,len(labels)):
#     fig, ax = plt.subplots(1, 2, figsize=(20, 10))
#     ax[0].imshow(np.squeeze(images_original[ix]))
#     ax[0].set_title('{0}'.format(ix+1), fontsize=25);
#     ax[1].imshow(np.squeeze(labels[ix]))
#     plt.show()     
print('Creating test image labels...')
test_labels = create_labels_noStat_noPrint(test_images_original)


"""
Augmentation (OPTIONAL)
"""
#augmentation() from augmentation.py
#Both image and label augmentation is done at the same time
#Augmented images and labels are added to the original datasets (but not saved in disk)
# images_augmented, labels_augmented = augmentation_Color(images_original, labels)
# images_augmented, labels_augmented = augmentation_Invert(images_original, labels)

"""
Grayscale images (unused)
"""
# #rgb2gray() from mask.py
# #shape (ix, height, width, 1(grayscale))
# print('Converting RGB training images to grayscale...')
# images_gray = rgb2gray(images_augmented)
# # #normalizing to 255
# images_gray=images_gray*255
# print('Converting RGB test images to grayscale...')
# test_images_gray = rgb2gray(test_images_original)
# # #normalizing to 255
# test_images_gray=test_images_gray*255
# # model will still need input of an array with 4 dimensions
# images_all=images_gray[..., np.newaxis]
# test_images_all=test_images_gray[..., np.newaxis]
# labels_all = np.concatenate((labels, labels), axis=0)

"""
Random color (unused)
"""
# #rgb2random() from mask.py
# #shape (ix, height, width, 3(random rgb))
# print('Randomly coloring RGB training images...')
# images_color = rgb2random(images_augmented)
# images_all = np.concatenate((images_augmented, images_color),axis=0)
# labels_all = np.concatenate((labels_augmented, labels_augmented), axis=0)
# #normalize pixel values. Maximum pixel value is 255.
# # images=images/255
# print('Randomly coloring RGB test images...')
# test_images_color = rgb2random(test_images_augmented)
# test_images_all = np.concatenate((test_images_augmented, test_images_color),axis=0)
# test_labels_all = np.concatenate((test_labels_augmented, test_labels_augmented), axis=0)
# #normalize pixel values. Maximum pixel value is 255.
# # test_images=test_images/255


"""
Edit masks (unused)
"""
# # if augmentation will only invert the images, masks need to be doubled and not changed
# masks = np.concatenate((masks,masks),axis=0)
# test_masks = np.concatenate((test_masks,test_masks),axis=0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
MODEL TRAINING

It takes images (n_img, h, w, 3(rgb)) and labels (n_img, h, w, n_classes) for training
Output has shape (n_img, h, w, n_classes)
Then it is converted and represented as a label (n_img, h, w, 3(rgb)) that can be visualized

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from models import unet, weighted_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from load_dataset import get_weights

##############################################################################

#HYPERPARAMETERS

#-Model-----------------------------------------------------------------------
split=0.21 # Validation and training dataset split (default 0.0 --> 0%)
pretrained_weights = None #'C://Users/Aritz.LEBERRI/Desktop/Project/Models/Sequences/123456.h5' # (default None)
input_size = (IMG_HEIGHT, IMG_WIDTH, 1) # (default (256,256,3)) (256,256,1) if grayscale
#get_weights() calculates weights with class frequencies in the dataset
#it takes the frequencies from both the training and test set

#getting weights after randomly coloring images gives issues
weights = get_weights(images_original,test_images_original) # (default [1.0, 1.0, 1.0, 1.0])
activation = 'elu' # (default 'relu') Activation on last layer is always softmax
dropout = 0.18 # (default 0.0)
dilation_rate = 1 # (default (1,1))
loss = weighted_categorical_crossentropy(weights) # (default 'categorical crossentropy')
reg = 0.01 # (default 0.01) L2 regularization
# learning_rate =  0.0001 
optimizer = 'adadelta' # (default 'adam')
#Try more or less layers
#Try w and w/o Batchnormalization()

#-Fit-----------------------------------------------------------------------
epochs = 100
batch_size = 1
callbacks=[EarlyStopping(patience=16,verbose=1),\
            ReduceLROnPlateau(factor=0.1, patience=5, min_delta=0.001, min_lr=0.0000001,verbose=1),\
                ModelCheckpoint('modelName.h5'.format(epochs),verbose=1, save_best_only=True,\
                                save_weights_only=False),\
                    CSVLogger('modelName.log')] #DON'T OVERWRITE
    #module 'h5py' has no attribute 'Group' <-- if folder does not exist
    
#-----------------------------------------------------------------------------

model = unet(pretrained_weights = pretrained_weights, input_size = input_size, weights = weights,\
         activation=activation, dropout=dropout, loss=loss, optimizer=optimizer, dilation_rate=dilation_rate)

#for array dataset with shape (n_img, size, size) one more dimension needs to be created in order to train
images_all=images_original[..., np.newaxis]
test_images_all = test_images_original[..., np.newaxis]
#normalize (not very effective if e.g. in one image maximum is 35000 and in another 15000. Background (and everything else) will be different)
# normalization_value = 255
# images_all = images_all/images_all.max()*normalization_value
# test_images_all = test_images_all/test_images_original.max()*normalization_value

results = model.fit(images_all, masks, validation_split=split, epochs=epochs, batch_size=batch_size,\
                      callbacks=callbacks, shuffle=True) 
print('Model correctly trained and saved')    
    
plt.figure(figsize=(8, 8))
plt.grid(False)
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
plt.grid(False)
plt.title("Learning curve ACCURACY", fontsize=25)
plt.plot(results.history["accuracy"], label="Accuracy")
plt.plot(results.history["val_accuracy"], label="Validation Accuracy")
plt.plot( p, results.history["val_accuracy"][p], marker="x", color="r", label="best model")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.legend();
plt.savefig(TEST_PREDICTIONS_PATH+'Accuracy')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL (repeated in the beginning of load.py)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from mask import output_to_label, get_max_in_mask
import matplotlib.patches as mpatches

##############################################################################

print('Testing on {0} images'.format(len(test_images_all))) 
test_outputs = model.predict(test_images_all, verbose=1)
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
ix_random_training = random.randint(0, len(images_all)-1)
image_all=images_all[ix_random_training][np.newaxis, ...]
train_outputs = model.predict(image_all, verbose=1)
print('Creating predicted label of training image {0}...'.format(ix_random_training+1))
train_outputs_labels=output_to_label(train_outputs)
fig, ax = plt.subplots(1, 3, figsize=(20, 10))
ax[0].imshow(np.squeeze(image_all), cmap="gray") #, cmap="gray" if grayscale
ax[0].set_title('Training image {0}'.format(ix_random_training+1), fontsize=25);
ax[0].set_xlabel('pixels', fontsize=16)
ax[0].set_ylabel('pixels', fontsize=16)
ax[1].imshow(np.squeeze(labels[ix_random_training]))
ax[1].set_title('Label', fontsize=25);
ax[1].set_xlabel('pixels', fontsize=16)
ax[1].set_ylabel('pixels', fontsize=16)
ax[2].imshow(np.squeeze(train_outputs_labels))
ax[2].set_title('Predicted label', fontsize=25);
ax[2].set_xlabel('pixels', fontsize=16)
ax[2].set_ylabel('pixels', fontsize=16)
plt.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.15), fontsize=18,\
           handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4)
plt.savefig(TEST_PREDICTIONS_PATH+'Train_{0}'.format(ix_random_training+1))
plt.show()

# Checking and saving ALL TEST samples
for ix in range (len(test_outputs)):
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].grid(False)
    ax[0].imshow(np.squeeze(test_images_all[ix]), cmap="gray")#, cmap="gray" if grayscale
    ax[0].set_title('Test image {0}'.format(ix+1), fontsize=25);
    ax[0].set_xlabel('pixels', fontsize=16)
    ax[0].set_ylabel('pixels', fontsize=16)
    ax[1].grid(False)
    ax[1].imshow(np.squeeze(test_labels[ix]))
    ax[1].set_title('Label', fontsize=25);
    ax[1].set_xlabel('pixels', fontsize=16)
    ax[1].set_ylabel('pixels', fontsize=16)
    ax[2].grid(False)
    ax[2].imshow(test_outputs_labels[ix])
    ax[2].set_title('Predicted label', fontsize=25);
    ax[2].set_xlabel('pixels', fontsize=16)
    ax[2].set_ylabel('pixels', fontsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.15), fontsize=18,\
                handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4)
    plt.savefig(TEST_PREDICTIONS_PATH+'Test{0}'.format(ix+1))
    plt.show()
    
#Model evaluation   
evaluation = model.evaluate(test_images_all, test_masks, batch_size=16)
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