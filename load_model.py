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
from load_dataset import *
from mask import *
import matplotlib.patches as mpatches
from models import weighted_categorical_crossentropy
import keras.losses

#############################################################################

"""
LOAD THE MODEL AND DATASET
"""
model_name='modelName' 
MODEL_PATH='/{0}.h5'.format(model_name)
custom_loss = keras.losses.penalized_loss = weighted_categorical_crossentropy()
model = load_model(MODEL_PATH, custom_objects={'wcce': custom_loss})
print ('Model correctly loaded')

# TRAIN_PATH = 'C://Users/Aritz.LEBERRI/Desktop/Project/Images/Train/7.All/' #training images dataset path
# TEST_PATH  = 'C://Users/Aritz.LEBERRI/Desktop/Project/Images/Test/' #testing images dataset path
TEST_PREDICTIONS_PATH = '' #testing outputs path
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS=4 #number of classes

"""
Load original dataset PNG IMAGES (old)
"""
# # load_images() from load_dataset.py
# images_original, test_images_original = load_images(TRAIN_PATH, TEST_PATH, TEST_PREDICTIONS_PATH, IMG_WIDTH, IMG_HEIGHT)

"""
Load original dataset ARRAYS (real pixel values)
"""
images_original = np.load('training_data.npy') 
test_images_original = np.load('test_data.npy') 
print('Dataset arrays correctly loaded')

"""
Create masks
"""
#create_masks() from mask.py
#shape (ix, height, width, n_classes)
# print('Creating training image masks...') 
# masks = create_masks(images_original)
print('Creating test image masks...')
test_masks = create_masks(test_images_original)

"""
Create labels
"""
#create_labels() from mask.py. Used for representation, but not for the model training
#shape (ix, height, width, 3), like images 
print('Creating training image labels...')
labels = create_labels(images_original)
# To check the labels
# for ix in range(0,len(labels)):
#     fig, ax = plt.subplots(1, 2, figsize=(20, 10))
#     ax[0].imshow(np.squeeze(images[ix]))
#     ax[0].set_title('{0}'.format(ix+1), fontsize=25);
#     ax[1].imshow(np.squeeze(labels[ix]))
#     plt.show()     
print('Creating test image labels...')
test_labels = create_labels_noStat_noPrint(test_images_original)

"""
Random color
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
# test_images_original = rgb2random(test_images_original)
# test_images_all = np.concatenate((test_images_augmented, test_images_color),axis=0)
# test_labels_all = np.concatenate((test_labels_augmented, test_labels_augmented), axis=0)
# #normalize pixel values. Maximum pixel value is 255.
# # test_images=test_images/255

"""
Grayscale images
"""
# #rgb2gray() from mask.py
# #shape (ix, height, width, 1(grayscale))
# print('Converting RGB training images to grayscale...')
# images_gray = rgb2gray(images_original)
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
Edit masks
"""
#if augmentation will only invert the images, masks need to be doubled and not changed
# masks = np.concatenate((masks,masks),axis=0)
# test_masks = np.concatenate((test_masks,test_masks),axis=0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#for array dataset with shape (n_img, size, size) one more dimension needs to be created in order to train
images_all=images_original[..., np.newaxis]
test_images_all = test_images_original[..., np.newaxis]
#normalize (not very effective if e.g. in one image maximum is 35000 and in another 15000. Background (and everything else) will be different)
# normalization_value = 255
# images_all = images_all/images_all.max()*normalization_value
# test_images_all = test_images_all/test_images_original.max()*normalization_value

print('Testing on {0} simulated images'.format(len(test_images_all))) 
test_outputs = model.predict(test_images_all, verbose=1)
print('Creating the predicted labels of the simulated test images...')
test_outputs_labels = output_to_label(test_outputs)
 
#Legend 1
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
print('The accuracy of the model on the simulated test set is: ',\
      evaluation[1]*100,'%')
print('The loss of the model on the simulated test set is: ',\
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


# ============================================================================
# CHECKING THE REAL TEST IMAGE (.FITS FILE)
# ============================================================================

# ONE REAL TEST IMAGE
#process_fits() from load_dataset
print('Loading real test image from fits file...')

size=256 #default size=256
normalized='no' #default normalized='yes'
normalization_value = 255 #default normalization_value=255
name = 'name'
image_data_use, test_images_real, details = process_fits(name='/{0}.fits'.format(name), size=size, \
                                                          normalized=normalized, normalization_value=normalization_value)
print('> Test image {0}'.format(name))
test_images_real=test_images_real[..., np.newaxis] #4 dimensions needed in order to be able to predict
print('  Testing on {0} real image sections'.format(len(test_images_real))) 
test_outputs_real = model.predict(test_images_real, verbose=1)

# images_small2big() from load_dataset to join all sections into a whole image again
# from mask input of (n_sections, size, size, 4) gives mask output of (size, size, 4)
test_outputs_real_big = images_small2big(images=test_outputs_real, details=details) 
test_outputs_real_big=test_outputs_real_big[np.newaxis, ...]
unique_elements_real, percentages_real = percentage_result(test_outputs_real_big)

#Legend 2
real_percentages = np.zeros(4)
for i in range (0, len(percentages_real)):
    real_percentages[int(unique_elements_real[i])] = percentages_real[i]

Background_percentage = mpatches.Patch(color=[0./255, 0./255, 0./255], label='Background: {0} %'.format(real_percentages[0]))
Glowing_percentage = mpatches.Patch(color=[0.35,1.,0.25], label='Glowing: {0} %'.format(real_percentages[1]))
Hot_pixel_percentage = mpatches.Patch(color=[0,0.5,1.], label='Hot pixel: {0} %'.format(real_percentages[2]))
Cluster_percentage = mpatches.Patch(color=[1, 0.2, 0.2], label='Cluster: {0} %'.format(real_percentages[3]))
    
# Check the ones with clusters in small sections
check_one_object(test_outputs_real, test_images_real, object_to_find='hot pixel', real_percentages=real_percentages, details=details)
# object_to_find options: 'background', 'glowing', 'hot pixel', 'cluster'

# Checking and saving THE TEST sample grayscale
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].grid(False)
ax[0].imshow(image_data_use, cmap="gray")
ax[0].set_title('Real test image', fontsize=25);
ax[0].set_xlabel('pixels', fontsize=16)
ax[0].set_ylabel('pixels', fontsize=16)
ax[1].grid(False)
ax[1].imshow(output_to_label(test_outputs_real_big)[0])
ax[1].set_title('Predicted label', fontsize=25);
ax[1].set_xlabel('pixels', fontsize=16)
ax[1].set_ylabel('pixels', fontsize=16)
plt.legend(loc='upper center', bbox_to_anchor=(0.15, -0.09), fontsize=16,\
    handles=[Background_percentage, Glowing_percentage, Hot_pixel_percentage, Cluster_percentage], ncol=4)    
plt.savefig(TEST_PREDICTIONS_PATH+'Real_Test_{0}_{1}'.format(model_name, name))
plt.show()

# Checking and saving THE TEST sample non grayscale
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].grid(False)
ax[0].imshow(image_data_use)
ax[0].set_title('Real test image', fontsize=25);
ax[0].set_xlabel('pixels', fontsize=16)
ax[0].set_ylabel('pixels', fontsize=16)
ax[1].grid(False)
ax[1].imshow(output_to_label(test_outputs_real_big)[0])
ax[1].set_title('Predicted label', fontsize=25);
ax[1].set_xlabel('pixels', fontsize=16)
ax[1].set_ylabel('pixels', fontsize=16)
plt.legend(loc='upper center', bbox_to_anchor=(0.15, -0.09), fontsize=16,\
    handles=[Background_percentage, Glowing_percentage, Hot_pixel_percentage, Cluster_percentage], ncol=4)    
plt.savefig(TEST_PREDICTIONS_PATH+'Real_Test_{0}_{1}'.format(model_name, name))
plt.show()
