# Image Simulation

## Introduction

In order to develop a ML model, data is vital. Frequently, collecting suitable data is more troublesome than writing algorithms. Depending on the sophistication of the problem, the number of parameters and the amount of data needed varies significantly. Most commonly used datasets for computer vision can provide over 100000 images, sometimes even a million of them. However, possessing such a vast and appropriate dataset is not always possible. Occasionally, data collection is overly costly, and the lack of data limits the model.

This obstacle is overcome creating images that simulate the ones taken by CCDs. The aim is to emulate the signatures of the four main categories to discriminate, so that the model will learn and subsequently be able to predict them on real detector images. To create a simulated image, first the background is defined by a fixed intensity value, or pedestal, to which an amount of noise can be added. Then the other three main categories to discriminate are added to the image: glowing, hot pixels and pixel clusters. These three differ in shape and pixel intensity (being pixel clusters the least intense, and glowing and hot pixels the most intense). The freedom of choice is the main advantage of this solution; it is possible to vary which objects (or categories) are included in a picture, the amount of them, their pixel intensities, size, position, etc. This way, a wide variety of simulated images can be used, ensuring that the model learns all kind of signals that can appear in a real CCD image.

## Image Simulation

The simulated images contain four classes/categories to segment: background, glowing, hot pixels and pixel clusters. Two datasets are created: a training and a testing set. The training set is used to train the ML model, while the testing set is only used once the model was fully trained. Trying to recreate real DAMIC images at T=140K as accurate as possible, specific pixel intensities are assigned to each class in 256X256 pixel images.

<p align="center">
<img src="https://github.com/aritzLizoain/CNN-Image-segmentation/blob/master/Images/Example_Images/Simulated_CCD_Image.png" width="400"/>
</p>

*Simulated 256X256 pixel CCD image containing glowing, hot pixels and pixel clusters. The pixel intensity values are given in ADCs.*

The background intensity, also referred as pedestal, is set a value of 8800ADC (Unit of measurement for charge in count of ADC through the digital output of the A/D con-verter) with a noise of ±60ADC, meaning that the background pixel intensities take values between 8740 and 8860 ADC (all pixel intensity ranges follow a gaussian distribution). Glowing is added as a vertical column with an intensity above background of range [1400, 1500]ADC. Hot pixels are added as thin vertical and horizontal lines of different lengths with intensities above background of 2200ADC, being the class with the highest pixel intensity value. Clusters, on the other hand, are added from a file containing the intensity and position of 803 pixel clusters. The file does not contain any alpha particle, hence almost all clusters are low-energy events and their intensity value is slightly above background.

Both python files containing the code for creating the simulated images, as well as the file with cluster information have been provided by Agustín Lantero Barreda, PhD Student of DAMIC-M.


