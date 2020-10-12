# Machine Learning

*Field of study that gives the ability to the computer to self-learn without being explicitly programmed.* Arthur L. Samuel, 1959.

## Introduction

As data growth continues to escalate, algorithms must keep improving to be able to understand it all with higher speed and higher accuracy. Processing the high amount of data that is produced every day is becoming more challenging without the assistance of ML. ML is an AI subset which is focused on developing programs that are able to teach themselves to make accurate predictions when exposed to new data. It is found in diverse sectors, due to its wide usage in image recognition, speech recognition, medical diagnoses, trading, etc. There are three main types of ML: supervised, unsupervised and reinforcement learning.

In supervised learning, algorithms are trained with labeled data. Thus, known input and output is being used. Each image data is tagged with its corresponding label, which is the desired output. The algorithm learns by comparing its prediction with the given label, and modifies the model accordingly. In this project classification supervised learning is applied, meaning that the output variables are categories ('background', 'glowing', 'hot pixels', and 'pixel clusters').

## Convolutional Neural Networks (CNNs)

Taking the human brain as a reference, artificial NNs are based on connected nodes called neurons. A neuron receives inputs from other neurons and combines them together. The output from a neuron is obtained by a value transformation called activation function. The values used in the activation function, termed weights, are randomly initialized. The accuracy of a neuron output is determined by the loss function; the lower the loss, the higher the output accuracy. Therefore the goal is to find the right weights to minimize the loss function, thus, giving the most accurate prediction. This is done by an optimizer, that identifies which weights contribute most directly to the loss of the network, subsequently updating them in order to minimize such loss. The training process is repeated for a number of iterations, aiming to improve the weight value readjustment.

CNNs are composed of an input layer, several hidden layers, and an output layer. Their employment allows the recognition of specific properties of image data, thereby becoming highly suitable for computer vision applications. Images are passed through the NN as an array of values describing pixel intensities. Each of these values is a feature that characterizes the image. The first few neuron layers learn low-level features (basic elements such as edges and colors), leading to a more complex pattern learning by the succeeding layers. This way, the network is able to differentiate one image from another. Generally, prediction accuracy is improved with a deeper network, i.e. with more layers.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/ConvNN.png" width="500"/>
</p>

*CNN feature learning process. Adapted Digital Image. Torres, J. "Convolutional Neural Networks for Beginners". (Towards Data Science, 2018). [Link](https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca).*

### Convolution

CNNs are named after its most important layer, the convolution layer. While a standard NN layer applies its activation function weights to the whole image, a convolution layer applies a set of weights spatially across the image, thereby reducing the number of parameters needed. This set of activation function weights compose the filter, which is defined by several hyper-parameters: filter size, stride, and depth. Filter size sets the width and height of the filter. The number of pixels to move before applying the filter again is set by stride. If the stride is smaller than the filter size, regions of the image are overlapped. The depth defines the number of channels of the filter, which is equal to the number of input channels (e.g. for a RGB image the depth is 3, one for each color channel, while for a grayscale image the depth is 1).

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Convolution.gif" width="450"/>
</p>

*Convolutional layer example. Saha, S. "A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way." Digital Image. (Towards Data Science, 2018). [Link](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53).*

### Max Pooling

Max pooling layers, like convolution layers, apply a filter across the image, which is also defined by a filter size and stride. The layer takes the maximum value within the filter, reducing the spatial size of the input. However, it does not take the maximum value across different depths, since it is applied to each depth channel individually.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Maxpool.gif" width="450"/>
</p>

*Max Pooling layer example. "coursera-deeplearning-ai-c4-week1." Digital Image. (Vernlium, 2018). [Link](https://vernlium.github.io/2018/10/15/coursera-deeplearning-ai-c4-week1/).*


### Dropout

Overfitting is one of the most common issues when training a ML model. It causes the model to memorize the training data, instead of learning from it, which leads to a high accuracy on the predictions while training, but a low accuracy on testing predictions. The most effective solution is adding more training data. Nonetheless, adding a dropout layer also helps avoiding the issue. Dropout layers randomly ignore a number of neuron outputs, reducing the dependency on the training set. 

<pre>
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Overfitting.png" width="400"/>           <img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Dropout.gif" width="400"/> 
</pre>

* LEFT: Overfitting representation. Adapted Digital Image. Despois, J. "Memorizing is not learning!". (Hackernoon, 2018). [Link](https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42). <br/> RIGHT: Dropout layer example. Digital Image. "https://mlblr.com/includes/mlai/index.html". (MLBLR). [Link](https://mlblr.com/includes/mlai/index.html).*


### Fully Connected

A fully connected layer is usually added as the last layer of the CNN. Like in a standard NN layer, every neuron is connected to every neuron in the previous layer. The fully connected layer classifies the image based on the outputs of the preceding layers.

## Image Segmentation

An image classification problem consists in predicting the object within the image. On the other hand, image segmentation requires a higher understanding of the image; the algorithm is expected to classify each pixel in the image. Thus, the output is a labeled image in which each pixel is classified to its corresponding category. Self-driving cars development, medical images diagnosis and satellite image analysis are some of the numerous image segmentation applications.
<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Image_label_representation.png" width="600"/>
</p>

*Image label example, where each pixel is classified as a class and its corresponding depth channel takes value 1. Adapted Digital Image. Jordan, J. "An overview of semantic image segmentation". (2018). [Link](https://www.jeremyjordan.me/semantic-segmentation).*


### U-Net

It is a widely held belief that a successful deep network training requires thousands of labeled training data. However, the U-Net network architecture, originally developed for biomedical (cell) image processing, uses the available labeled images more efficiently. In consequence, it has become one of the most popular networks in the medical domain, where usually thousands of training samples are beyond reach. Furthermore, the U-Net architecture is able to detect small size objects within the image.

The U-Net architecture contains two paths: the contraction path, called encoder, and its symmetric expanding path, the decoder. The encoder is built stacking convolutional and max pooling layers. This way the size of the image is reduced, which is called down sampling. The deeper the network, the more reduced is the image. This allows obtaining information about the objects within the image, but the spatial information is lost. Therefore the image needs to be up sampled to the original image size, i.e. restore the low resolution image to a high resolution image. For this purpose, the decoder is built stacking convolutional and transposed convolutional layers. Every step of the decoder uses skip connections by concatenating the outputs of the down sampling layers with the up sampling layer at the corresponding level. 
The network does not contain fully connected layers, therefore is defined as a fully convolutional network.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Image_label_representation.png" width="600"/>
</p>

*Image label example, where each pixel is classified as a class and its corresponding depth channel takes value 1. Adapted Digital Image. Jordan, J. "An overview of semantic image segmentation". (2018). [Link](https://www.jeremyjordan.me/semantic-segmentation).*

