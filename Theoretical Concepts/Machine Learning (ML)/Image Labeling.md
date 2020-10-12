# Machine Learning (ML): Image Labeling

## Introduction

In supervised learning image labels are as essential as images. A label represents the desired output, i.e. an image where each pixel is classified to its corresponding class. A model learns by comparing its prediction with the label and trying to minimize the number of incorrectly predicted pixels.

## Automated Labeling

A new method of image labeling is studied, straightforwardly without the employment of any labeling platform or annotation tool. Reflecting on how the simulated images are created, a great convenience is found: the images are created with predetermined pixel intensities. Thus, the threshold values that determine whether a pixel belongs to a certain class are known. This knowledge plays a decisive part in the development of an automated labeling method. 

The datasets are three-dimensional arrays of shape (number of images, height, width). The goal is to obtain a four-dimensional array of shape (number of images, height, width, 4), where the last dimension informs about the class corresponding to the pixel. This way, the label is a segmentation map where each pixel contains a class label represented as an integer; numbers 1, 2, 3 and 4 are respectively assigned to the classes 'background', 'glowing', 'hot pixel' and 'cluster'. These class labels are described in a one-hot encoded way, meaning that each one has a depth channel (e.g. [0,0,1,0] represents the class 'hot pixel').

To get started, a blank four-dimensional array is created. Then, going through all the images, each pixel is classified. Knowing the threshold values of the four categories, the pixel intensity value determines its class label. Based on this classification, the corresponding depth channel takes value 1 while the other three remain at value 0. Going through all the images, the segmentation maps are obtained.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Image_label_representation.png" width="400"/>
</p>

*Image label example, where each pixel is classified as a class and its corresponding depth channel takes value 1. Adapted Digital Image. Jordan, J. "An overview of semantic image segmentation". (2018). [Link](https://www.jeremyjordan.me/semantic-segmentation).*

Labeling the images in an automated way saves a substantial amount of time. Additionally, the knowledge of the threshold values of each class allows correctly classifying every pixel, thereby obtaining an exact label.

In order to visualize the label, the array needs to be reshaped into (number of images, height, width, 3(RGB)). First, the position of the maximum value on the fourth dimension (i.e. the integer assigned to the class) is taken. This shows the regions of the image where each class is present. Lastly, a different color multiplier (a three-dimensional RGB array) is applied to each class.

\begin{figure}
\centering
\includegraphics[scale=0.7]{SegmentationMap.png}
\caption[Segmentation map]{Segmentation map example. Adapted Digital Image. Jordan, J. \emph{An overview of semantic image segmentation}. (2018). \url{https://www.jeremyjordan.me/semantic-segmentation}.}
\label{fig:5}
\end{figure}

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Segmentation_map.png" width="400"/>
</p>

*Segmentation map example. Adapted Digital Image. Jordan, J. "An overview of semantic image segmentation". (2018). [Link](https://www.jeremyjordan.me/semantic-segmentation).*
