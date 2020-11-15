# -*- coding: utf-8 -*-
"""
//////////////////////////////////////////////////////////////////////////////////////////
// Original author: Agustín Lantero Barreda (PhD Student of DAMIC-M)
// Modified by: Aritz Lizoain
// Github: https://github.com/aritzLizoain
// My personal website: https://aritzlizoain.github.io/
// Description: CNN Image Segmentation
// Copyright 2020, Aritz Lizoain.
// License: MIT License
//////////////////////////////////////////////////////////////////////////////////////////

Working directory must be where all files are located
This code can be run to create simulated images
Image details can be changed in image_details.py
"""

import image_details
import matplotlib.pyplot as plt
import numpy as np

#############################################################################

image_name=''
SAVING_PATH='/{0}'.format(image_name)
amountOfImages=200
size=256
#size2= if square image is not wanted

#random nSamples
r=round(np.random.uniform(300,500))
#print("Number of samples is: {0}".format(r))
nSamples = 10
#random noise
# r=round(np.random.uniform(0,0))
noise = 60 #0.1852*np.sqrt(400)
#print("Amount of noise is: {0}".format(r))

image_array = np.zeros((amountOfImages, size, size)) 
image = image_details.image_details(size,size,0,0,noise,nSamples,0/24) #5e-2,0
for i in range(0, amountOfImages): 
### Simulated the dark current noise
    image.darkCurrent
### Add random (and random amount) clusters to the image 
    r=round(np.random.uniform(5000,5000))
    #print("Number of clusters of sample {1} is: {0}".format(r, i))
    image.add_cluster(r)
### Add random (and random amount) hot pixels to the image
    #Usual number of hot pixels is: 10 in a 4000X4000 pixel image
    r=round(np.random.uniform(400,500))
    #print("Number of hot pixels is: {0}".format(r))
    image.add_hotPixels(r)
### Add glowing to the first part of the iamge
    #Usual width of glowing is: 200 in a 4000X4000 pixel image
    
    #Glowing and no glowing randomly
    gl=round(np.random.uniform(0,4))
    if gl > 2: #2/5 chances for having no glowing
        r=round(np.random.uniform(0,0))
    else:
        r=round(np.random.uniform(40,61))
    r2=round(np.random.uniform(1400,1500))
    g=round(np.random.uniform(0,4))
    #print("Width and intensity of glowing are: {0} & {1}".format(r, r2))
    image.add_glowing(r,r2, g)
### Simulates the readout noise
    image.read_image
    # plt.figure(figsize=(size/100,size/100), dpi=132.68) #defining output size. dpi might need to change +-0.05 
    # plt.figure(figsize=(5.52,5.52), dpi=100) #defining output size
    ## Plot the region [xmin,xmax]x[ymin,ymax]
    # image.plot_image(0,size,0,size) 
    # plt.savefig(SAVING_PATH+'{0}'.format(i+1), bbox_inches='tight', transparent="True", pad_inches=0) #REMOVING PADDING
### Plot the pixel charge distribution
    # image.plot_chargeDist(400, r"Ineff = $10^{-6}$")
    #plt.savefig("PixChargeDist.png")
### Save image array. Real pixel values.
    image_array[i] = image.image

np.save('training_data.npy', image_array)
# image_array_loaded = np.load('training_data.npy') # TO LOAD  
