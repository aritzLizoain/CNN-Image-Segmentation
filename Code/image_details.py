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

Details can be changed to create images in image_simulation.py
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle

#############################################################################

class image_details(object):
      ''' Simulate a CCD Image'''
      def __init__(self, nx, ny, darkC, pedestal, noise, nSamples, texp):
          self.nx, self.ny = nx, ny
          pedestal = 8800
          self.darkC, self.noise, self.pedestal = darkC, noise, pedestal
          self.nSamples, self.texp = nSamples, texp
          self.day_pix = 1/(24*12e7) #pix/day
          
          
      @property
      def darkCurrent(self):
          ''' Simulate Random Poisson electrons as Dark Current'''
          self.readout_time
          lambd = 0#self.darkC*(self.texp+self.tread)
          #print("Lambda = ", str(lambd))
          self.image = 0#np.random.poisson(lambd, (self.ny,self.nx))

      @property 
      def readout_time(self):
          ''' Time each pixel remains without being read'''
          ''' t = time/pix x (nxi + (Nx+1)nyi + 1) x Nsamples '''
          if self.texp == 0:
              ''' There is no difference between exposure of pixels (Continous readout)'''
              self.tread = self.day_pix*( self.nx + self.ny*(self.nx+1) )*self.nSamples
              #print("Readout time {0} hours".format(self.tread*24))
          else:
              n_pix = np.arange(0, self.nx, 1)
              npix_matrix = np.tile(n_pix, [self.ny,1]) + (self.nx + 1)*np.array([[i] for i in range(self.ny)]) + 1
              self.tread = self.day_pix*npix_matrix*self.nSamples 
              print("Readout time {0} hours".format(self.tread*24))             
             
      @property
      def read_image(self):
          '''Convolution of Poisson noise + Gaussian readout'''
          self.image = self.image + np.random.normal(self.pedestal, self.noise/np.sqrt(self.nSamples), (self.ny,self.nx)) 

      def reduce_image(self, scale):
            '''Reduce the image to 1xscale summing charge'''
            red_im = np.sum(self.image.reshape(self.ny, int(self.nx/scale),scale), axis = 2)
            self.image = red_im
            self.ny, self.nx = red_im.shape
            print("Image Shape {0}x{1}".format(self.nx, self.ny))

      def charge2ADU(self, e2ADU):
          '''Convert the image from electrons to ADU '''
          self.image = self.image*e2ADU

      def plot_chargeDist(self, nbins, lab, norm=False):
          ''' Plots the pixel Charge distribution of the image '''
          rep = self.nx*self.ny
          #scale = [396940/(3100*150)] * rep
          plt.hist(self.image.flatten(), nbins, density=norm, alpha=0.7, label=lab)#, weights = scale)
          plt.yscale('log')
          plt.xlabel(r"Charge $[e^-]$")
          plt.xlim(-0.5,1.5)
          #x_ticks = np.arange(0, np.round(np.amax(self.image)+1),1)
          #plt.xticks(x_ticks)
          #plt.show()

      def add_cluster(self, nClusters):
          ''' Add Clusters from pickle file'''
          f = open("Cluster.pkl", "rb")
          clusters = pickle.load(f)
          clusters = [i * 200 for i in clusters] #intensity of clusters multiplied by 200
          image_clust = np.zeros([self.ny, self.nx])
          randomNClusters=round(np.random.uniform(0,len(clusters)))  
          for clust in clusters[randomNClusters:randomNClusters+nClusters]:
              ### Cluster size
              ny_c, nx_c = clust.shape
              ### Random index 
              ny_r, nx_r = int(np.random.uniform(0, self.ny-ny_c)), int(np.random.uniform(144, 255-nx_c)) #changed x and y
              #145 comes from 255-109. 109 is the longest cluster size
              nyc_max, nxc_max = ny_c + ny_r, nx_c + nx_r 
              image_clust[ny_r:nyc_max,nx_r:nxc_max] = image_clust[ny_r:nyc_max,nx_r:nxc_max] + np.round(clust)
          self.image = self.image + image_clust
          #plt.imshow(image_clust)
          #plt.show()
         
      def add_hotPixels(self, nHotPixels):
          ''' Adds a row or column of hot pixels '''
          image_hotPix = np.zeros([self.ny, self.nx])
          for i in range(nHotPixels):
              ### Hot pixel initial coordinates
              ny_h, nx_h = int(np.random.uniform(0, self.ny)), int(np.random.uniform(66, 144)) #self.nx or self.ny
              rowCol = np.random.randint(2, size = 1)
              if rowCol == 0:
                ''' Row Hot pixels  '''  
                hotPix_len = int(np.random.uniform(nx_h,144)) #self.nx
                hotPixels = np.zeros([int(hotPix_len-nx_h)]) + 2200
                image_hotPix[ny_h, nx_h:hotPix_len] = hotPixels
              else:
                ''' Column Hot pixels  '''
                hotPix_len = int(np.random.uniform(ny_h,self.ny))
                hotPixels = np.zeros([int(hotPix_len-ny_h)]).T + 2200
                image_hotPix[ny_h:hotPix_len, nx_h] = hotPixels
          self.image = self.image + image_hotPix
       
      def add_glowing(self, nx, q_glow, g=0):
          '''Add glowing to the side near the detector ''' 
          glow_im = np.zeros([self.ny,self.nx])
          glow_im[:,g:nx+g] = np.zeros([self.ny,nx]) + q_glow # switched x and y. +2 so it does not touch the left side
          self.image = self.image + glow_im

      def plot_image(self, nx_min, nx_max, ny_min, ny_max):
          ''' Plots the image'''
          #mask_im = self.image[ny_min:ny_max,nx_min:nx_max] < 0.7
          #im2plot = np.ma.masked_array( self.image[ny_min:ny_max,nx_min:nx_max], mask = mask_im )
          #plt.imshow(self.image, aspect='auto')
          plt.imshow(self.image[ny_min:ny_max,nx_min:nx_max])
          #plt.show()
          #taking axis out
          plt.imshow(self.image, interpolation='nearest')
          plt.axis('off')
          plt.plot()
          #plt.title("Image [{0},{1}]x[{2},{3}]".format(nx_min, nx_max, ny_min, ny_max))
    
      # def transfer_charge(self, ineff):
      #     ''' Simulates charge transfer between pixels with inefficiency'''
      #     ### Number of times pixel is read per row (Row+Serial Reg)
      #     n_pix = np.arange(0, self.nx, 1)
      #     npix_matrix = np.tile(n_pix, [self.ny,1]) + np.array([[1+i] for i in range(self.ny)])
      #     matrix_loss = np.power(1-ineff, npix_matrix)
      #     image_gain = np.zeros([self.ny,self.nx])
      #     image_gain[:,1:] = np.cumsum(self.image*matrix_loss, axis = 1)[:,:-1]
      #     self.image = self.image*matrix_loss + ineff*image_gain

      def transfer_charge(self, CTI):
          '''Simulates charge transfer between pixels with inefficiency
          following a random Binomial distribution B(Npix, CTI)'''
          n_pix = np.arange(0, self.nx, 1)
          npix_matrix = np.tile(n_pix, [self.ny,1]) + np.array([[1+i] for i in range(self.ny)])
          ### Losses Matrix
          B = np.random.binomial(npix_matrix, CTI)
          ### Image Matrix
          I = self.image
          ### If loss > Charge. All the charge is lost
          B[I<=B] = I[I<=B]
          ### First Column never gain charge
          #I[:,1:self.nx] = I[:,1:self.nx] + B[:,:-1]
          print(I<0)
          self.image = I - B 
  
