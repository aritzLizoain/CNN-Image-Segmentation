# Network Implementation

Due to its capacity to work efficiently with a reduced amount of images and to detect small size objects, a U-Net structure is implemented. The structure is constituted by two main parts: the encoder and the decoder.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Unet.png" width="800"/>
</p>

*Implemented U-Net architecture. Adapted from Ronneberger, O. et al. "Convolutional Networks for Biomedical Image Segmentation". (2015). [Link](https://arxiv.org/pdf/1505.04597.pdf).*

The encoder, following a typical CNN architecture, consists of the repeated application of two 3X3 convolutional layers along with a 2X2 max pooling layer. The decoder, on the other side, consists of an iteration of a 2X2 transposed convolution, a concatenation with the down sampling layer at the corresponding level, and two 3X3 convolutional layers. The last layer is a 1X1 convolutional layer with a softmax activation function, which returns a four-dimensional vector with the output of the previous layer transformed into a probability distribution ranged between 0 and 1.

The total number of convolutional layers in this project is 19, in contrast to the 23 in the original architecture. The model is not able to properly train with the initial structure, due to the excessive depth; the images are overly down sampled and the model struggles to learn such small details. For this reason, one level of layers is removed from the structure, leading to a proper model training. 

## Activation Function

The implemented network also differentiates in the activation function (except for the last convolutional layer); the ELU (Exponential Linear Unit) is applied, instead of the ReLU (Rectified Linear Unit) function, in order to deal with the ‘dying ReLU (Rectified Linear Unit) problem’.
ReLU outputs zero for all negative inputs, meaning that if the network leads to negative inputs into a ReLU, the neuron is not contributing to the network’s learning, and its information is lost

<pre>
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Relu.png" width="400"/>           <img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Elu.png" width="400"/> 
</pre>

*LEFT: ReLU activation function. Digital Image. Ms. Karnam Shubha "Activation Functions". (360digitmg, 2020). [Link](https://360digitmg.com/activation-functions-neural-networks#relu).* <br/> *RIGHT: ELU activation function. Digital Image. Ms. Karnam Shubha "Activation Functions". (360digitmg, 2020). [Link](https://360digitmg.com/activation-functions-neural-networks#relu).*

## Loss Function

The original loss function, on the other side, is the categorical crossentropy. This function is used on multi-class classification applications, where the last activation function outputs a probability distribution vector. However, its employment does not allow obtaining meaningful predictions. This loss function gives the same importance to all classes, no matter how frequent they are in the dataset. Since the simulated set contained background pixels in its majority, this class has a significantly greater impact on the loss function. For this reason, in order to work with the imbalanced dataset, a weighted version of the loss function is implemented. With the weighted categorical crossentropy, the least frequent classes is given the highest weight, or importance, thereby balancing the impact of all classes on the loss function.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Crossentropy.png" width="500"/>
</p>

*Categorical CrossEntropy. Digital Image. "How to use binary & categorical crossentropy with Keras?". (Machinecurve, 2019). [Link](https://www.machinecurve.com/index.php/2019/10/22/how-to-use-binary-categorical-crossentropy-with-keras/).*

## Optimizer

The optimizer is not originally detailed, therefore the most common ones are tested: Adam, Adagrad, Adadelta and SGD. All of them are adapted stochastic gradient descent methods. These methods are the algorithms that change the weights of the activation function in order to reduce the loss given by the loss function. An optimizer is defined by its learning rate. This hyper-parameter determines the amount of weights that are updated at each training iteration. The larger the learning rate, the faster the optimizer will minimize the loss function. However, if the learning rate is too large, the optimizer might not be able to converge and minimize the loss function. In the end, the chosen optimizer is adadelta, a method which dynamically adapts its learning rate over time. The automatic learning rate setting is found to be highly convenient, and works efficiently on the simulated dataset.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Optimizers.gif" width="400"/>
</p>

*Optimization algorithm animations. Digital Image. "Alec Radford's animations for optimization algorithms". (2015). [Link](http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html).*

## Dropout 

Finally, unlike in the original structure, dropout layers are added between every pair of convolutional layers. The intention is to avoid overfitting as much as possible.


There exists no rule on which layers, activation function, loss function or optimizer work better, therefore various models have to be tested. Not only all parameters have to be tested, but even different combinations of them, in order to find out which ones work most efficiently on the given set. Considering that a 100 epoch training takes more than 5h (~3-4 minutes per epoch), the structure and parameter optimization of the model end up being especially time-consuming.
