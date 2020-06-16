# -*- coding: utf-8 -*-
"""
@author: Agustín

IMAGE SIMULATION

Working directory must be where all files are located: 
C://Users/Aritz/Desktop/Project

This code can be run to create simulated images
Image details can be changed in image_details.py

"""

import image_details
import matplotlib.pyplot as plt
import numpy as np

#############################################################################

image_name=''
SAVING_PATH='C://Users/Aritz/Desktop/Project/Images/Train/{0}'.format(image_name)
amountOfImages=1
size=256
#size2= if square image is not wanted

#random nSamples
r=round(np.random.uniform(300,500))
#print("Number of samples is: {0}".format(r))
nSamples = 1
#random noise
r=round(np.random.uniform(0,0))
noise = 0#0.1852*np.sqrt(400)
#print("Amount of noise is: {0}".format(r))

image = image_details.image_details(size,size,0,0,noise,nSamples,0/24) #5e-2,0
for i in range(0, amountOfImages): 
### Simulated the dark current noise
    image.darkCurrent
### Add random (and random amount) clusters to the image 
    r=round(np.random.uniform(5,12))
    #print("Number of clusters of sample {1} is: {0}".format(r, i))
    image.add_cluster(r)
### Add random (and random amount) hot pixels to the image
    #Usual number of hot pixels is: 10 in a 4000X4000 pixel image
    r=round(np.random.uniform(2,6))
    #print("Number of hot pixels is: {0}".format(r))
    image.add_hotPixels(r)
### Add glowing to the first part of the iamge
    #Usual width of glowing is: 200 in a 4000X4000 pixel image
    r=round(np.random.uniform(0,14))
    r2=round(np.random.uniform(10,30))
    #print("Width and intensity of glowing are: {0} & {1}".format(r, r2))
    image.add_glowing(r,10)
### Simulates the readout noise
    image.read_image
    plt.figure(figsize=(size/100,size/100), dpi=132.68) #defining output size. dpi might need to change +-0.05
    #plt.figure(figsize=(5.52,5.52), dpi=100) #defining output size
    ### Plot the region [xmin,xmax]x[ymin,ymax]
    image.plot_image(0,size,0,size) 
    plt.savefig(SAVING_PATH+'{0}'.format(i+1), bbox_inches='tight', transparent="True", pad_inches=0) #REMOVING PADDING
### Plot the pixel charge distribution
    #image.plot_chargeDist(400, r"Ineff = $10^{-6}$")
    #plt.savefig("PixChargeDist.png")
    




