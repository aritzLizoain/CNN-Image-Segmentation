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

- load_images (unused)
- get_weights: calculates the weights for the loss function
- process_fits: loads FITS files and creates small sections
- images_small2big: reconstructs small sections
- check_one_object: looks for the chosen category section by section 
"""

import os
import sys
import numpy as np
import cv2
from skimage.transform import resize

##############################################################

# NOT USED IN VERSION 2.0.
# THE IMAGES ARE NOW SAVED AND LOADED AS ARRAYS, NOT AS PNG FILES

# def load_images(TRAIN_PATH='', TEST_PATH='',\
#                 TEST_PREDICTIONS_PATH='',IMG_WIDTH = \
#                 256, IMG_HEIGHT = 256):

#     train_ids = next(os.walk(TRAIN_PATH))[2]
#     test_ids = next(os.walk(TEST_PATH))[2]

#     # Get and resize train images and masks
#     images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,3)\
#                       , dtype=np.uint8)
#     test_images = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH\
#                             , 3), dtype=np.uint8)
#     sys.stdout.flush()
    
#     # # train images 
#     for n,id_ in enumerate(train_ids):
#         img = cv2.imread(TRAIN_PATH + id_)
#         img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant'\
#                      , preserve_range=True)
#         images[n] = img
        
#         # # test images 
#         for n,id_ in enumerate(test_ids):
#             mask_ = cv2.imread(TEST_PATH + id_)
#             mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH),\
#                            preserve_range=True, mode='constant')
#             test_images[n] = mask_  
#     print('Dataset correctly loaded')
#     return images, test_images

#-------------------------------------------------------------

def get_weights(images,test_images):
    from mask import get_percentages
    #all_images = np.concatenate((images, test_images)) to take
    #both training and test images
    all_images=images #to take only training images
    unique_elements, percentage = get_percentages(all_images)
    inverse_percentages=1/percentage #the weights are inversely
    #proportional to their frequency
    weights = inverse_percentages/sum(inverse_percentages)*\
        len(unique_elements) #normalize to the number of classes
    return weights

#-------------------------------------------------------------

def process_fits(name='name.fits', size=256, normalized='yes'\
                 , normalization_value=255):
    import matplotlib.pyplot as plt
    from astropy.visualization import astropy_mpl_style
    plt.style.use(astropy_mpl_style)
    from astropy.utils.data import get_pkg_data_filename
    from astropy.io import fits
    import numpy as np
    
    #LOADING THE IMAGE AND GETTING INFORMATION
    image_file = get_pkg_data_filename(name)
    image_data = fits.getdata(image_file, ext=0)
    # image_data=image_data/100
        
    # normalize
    if normalized=='yes':
        maximum_value=np.amax(image_data)
        image_data_normalized=image_data/maximum_value*\
            normalization_value
    elif normalized=='no':
        # image_data=image_data
        None
    else:
        print('  ERROR: The given input for the normalization\
              variable is not an option. Please choose yes/no')
    
    #information about the original full image
    image_length=image_data.shape[1]
    image_height=image_data.shape[0]
    amount_images_wide=int((image_length/2)/size) #we will only
        #take half of the image
    amount_images_high=int(image_height/size)

    # # RESIZE image UNUSED
    # if image_length/size-amount_images_wide < 0.5:
    #     amount_images_wide=amount_images_wide
    # else:
    #     amount_images_wide=amount_images_wide + 1  
    # if image_height/size-amount_images_high < 0.5:
    #     amount_images_high=amount_images_high
    # else:
    #     amount_images_high=amount_images_high + 1
    
    # number_of_images=amount_images_wide*amount_images_high
       
    # if normalized=='yes':
    #     image_data_normalized_resized=np.resize(image_data_normalized, (size*amount_images_high, size*amount_images_wide))
    #     print('  Resized and normalized real test image shape: {0}'.format(image_data_normalized_resized.shape))
    #     plt.figure()
    #     plt.imshow(image_data_normalized_resized)
    #     plt.colorbar()
    #     plt.title('Normalized and resized real test image', fontsize=15)
    #     plt.show()
    #     image_data_use = image_data_normalized_resized
    # elif normalized=='no':
    #     image_data_resized=np.resize(image_data, (size*amount_images_high, size*amount_images_wide))
    #     print('  Resized real test image shape: {0}'.format(image_data_resized.shape))
    #     plt.figure()
    #     plt.imshow(image_data_resized)
    #     plt.colorbar()
    #     plt.title('Resized real test image', fontsize=25)
    #     plt.show()
    #     image_data_use = image_data_resized
    
    #CUT
    number_of_images = amount_images_wide*amount_images_high
    image_data_use=np.zeros((amount_images_high*size,amount_images_wide*size))
    starting_value=image_data.shape[1]-image_data_use.shape[1]
    if normalized=='yes':
        for i in range(0,image_data_use.shape[0]):
            for j in range (0,image_data_use.shape[1]):
                image_data_use[i,j] = image_data_normalized[i,j + starting_value]
        print('  Cut and normalized real test image shape: {0}'.format(image_data_use.shape))
        plt.figure()
        plt.grid(False)
        plt.imshow(image_data_use)
        plt.colorbar()
        plt.title('Normalized and cut real test image', fontsize=15)
        plt.show()
        
    elif normalized=='no':
        for i in range(0,image_data_use.shape[0]):
            for j in range (0,image_data_use.shape[1]):
                image_data_use[i,j] = image_data[i,j + starting_value]
        plt.figure()
        plt.grid(False)
        plt.imshow(image_data_use)
        plt.colorbar()
        plt.title('Cut real test image', fontsize=20)
        plt.show()
        print('  Cut real test image shape: {0}'.format(image_data_use.shape))
               
        
  # Create the smaller sections
    print('  Creating {1} sections of size {0}X{0}...'.format(size, number_of_images))
    images_small=np.zeros((number_of_images,size,size))
    # print('  Images small shape: {0}'.format(images_small.shape))
    for i in range(0, amount_images_wide):
        for j in range(0, amount_images_high):
            for x in range(0, size):
                for y in range (0, size):
                    images_small[i+j*(amount_images_wide),y,x]=image_data_use[y+j*size,x+i*size]              
    print('  Real test images correctly created')
    details=np.array([size, amount_images_high, amount_images_wide], dtype=int)
    
    return image_data_use, images_small, details

#----------------------------------------------------------------------------

# from mask input of (n_sections, size, size, 4) gives mask output of (size, size, 4)
def images_small2big(images, details):
    # Create the big image from small sections
    size = details[0]
    amount_images_high = details[1]
    amount_images_wide = details[2]
    dimensions = images.shape[3]
    full_image_empty = np.zeros((size*amount_images_high, size*amount_images_wide, dimensions))
    print('  Creating the real predicted test image from the {0} sections...'.format(len(images)))
    for i in range(0, amount_images_wide):
        for j in range(0, amount_images_high):
            for x in range(0, size):
                for y in range (0, size):
                    full_image_empty[y+j*size,x+i*size] = images[i+j*(amount_images_wide),y,x]            
    print('  Real test image prediction correctly created')
    return full_image_empty

#----------------------------------------------------------------------------
# CHECK THE ONES WITH A SPECIFIC OBJECT IN SMALL SECTIONS
def check_one_object(test_outputs_real, test_images_real, object_to_find='Cluster', real_percentages=[0,0,0,0], details=[0,0,0]):
    from mask import get_max_in_mask, mask_to_label
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
 
    if object_to_find=='Background':
        object_number = 0
    elif object_to_find=='Glowing':
        object_number = 1
    elif object_to_find=='Hot pixel':
        object_number = 2
    elif object_to_find=='Cluster':
        object_number = 3
    else:
        print('  ERROR: The given input for the object to find variable is not an option.\
              Please choose background/glowing/hot pixel/cluster')
    
    #Legend 1
    red_patch = mpatches.Patch(color=[1, 0.2, 0.2], label='Cluster')
    blue_patch = mpatches.Patch(color=[0,0.5,1.], label='Hot pixel')
    green_patch = mpatches.Patch(color=[0.35,1.,0.25], label='Glowing')
    black_patch = mpatches.Patch(color=[0./255, 0./255, 0./255], label='Background')
       
    counter = 0
    for i in range (len(test_outputs_real)):
        check=test_outputs_real[i]
        check=check[np.newaxis, ...]
        check=get_max_in_mask(check)
        is_there=object_number in check
        #in order to know the position of each section
        ychange = int(i/details[2])*details[0] #y axis position
        xchange = (i-int(i/details[2])*details[2])*details[0] #x axis position
        if is_there == True:
            from mask import output_to_label_one_object
            label_with_one_object = output_to_label_one_object(check, object_number)
            label_all_objects = mask_to_label(check, to_print='no')
            fig, ax = plt.subplots(1, 3, figsize=(20, 10))
            # plt.setp(ax, xticklabels=pixels, yticklabels=pixels)
            ax[0].grid(False)
            ax0 = ax[0].imshow(np.squeeze(test_images_real[i]))
            ax[0].set_title('Section {0}'.format(i+1), fontsize=25);
            ax[0].set_xlabel('pixels', fontsize=16)
            ax[0].set_ylabel('pixels', fontsize=16)
            ax[0].set_xticks([0,50,100,150,200,250])
            ax[0].set_xticklabels([0+xchange,50+xchange,100+xchange,150+xchange,200+xchange,250+xchange])
            ax[0].set_yticks([0,50,100,150,200,250]) 
            ax[0].set_yticklabels([0+ychange,50+ychange,100+ychange,150+ychange,200+ychange,250+ychange])
            cax = fig.add_axes([0.12, 0.16, 0.25, 0.03])
            plt.colorbar(ax0, orientation="horizontal", cax=cax)
            ax[1].grid(False)
            ax[1].imshow(label_all_objects[0])
            ax[1].set_title('Predicted label', fontsize=25);
            ax[1].set_xlabel('pixels', fontsize=16)
            ax[1].set_ylabel('pixels', fontsize=16)
            ax[1].set_xticks([0,50,100,150,200,250])
            ax[1].set_xticklabels([0+xchange,50+xchange,100+xchange,150+xchange,200+xchange,250+xchange])
            ax[1].set_yticks([0,50,100,150,200,250])
            ax[1].set_yticklabels([0+ychange,50+ychange,100+ychange,150+ychange,200+ychange,250+ychange])
            ax[2].grid(False)
            ax[2].imshow(label_with_one_object[0])
            ax[2].set_title('Finding {0}'.format(object_to_find), fontsize=25);
            ax[2].set_xlabel('pixels', fontsize=16)
            ax[2].set_ylabel('pixels', fontsize=16)
            ax[2].set_xticks([0,50,100,150,200,250])
            ax[2].set_xticklabels([0+xchange,50+xchange,100+xchange,150+xchange,200+xchange,250+xchange])
            ax[2].set_yticks([0,50,100,150,200,250])
            ax[2].set_yticklabels([0+ychange,50+ychange,100+ychange,150+ychange,200+ychange,250+ychange])
            plt.legend(loc='upper center', bbox_to_anchor=(2.1, 1.5), fontsize=16,\
                       handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4) 
            plt.show() #the image is not being saved
            counter=counter + 1 
            # print('  {1} found in section {0}'.format(i, object_to_find))
        else:
            counter=counter
    print('  {1} found in {0} sections'.format(counter, object_to_find))
    return None

