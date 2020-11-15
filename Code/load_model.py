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

This code can be run to load a model, predict labels and evaluate results
It processes FITS files and analyzes them by sections
"""

import numpy as np
import matplotlib.pyplot as plt
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

# ============================================================================
# CHECKING THE DAMIC TEST IMAGE (.FITS FILE)
# ============================================================================

# ONE REAL TEST IMAGE (a 'for' loop can be implemented in order to test more than one image)
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
# from input shaped (n_sections, size, size, 4) gives output shaped (size, size, 4)
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
