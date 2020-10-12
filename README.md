# CNN Image Segmentation
## Application of deep learning techniques to images collected with Charge Coupled Devices to search for Dark Matter


The Standard Model of particle physics, while being able to make accurate predictions, has been proved to fail to explain various phenomena, such as astronomical dark matter observations.
In this work, a machine learning application has been implemented with the goal of studying dark matter candidates.
Images from Charge Coupled Devices (CCDs) in different experiments DAMIC/DAMIC-M  located underground are used to test different deep learning algorithms.
A U-Net model is trained with Python's open-source library Keras. The model performs multi-class image segmentation in order to detect dark matter particle signals among background noise.

For more information regarding dark matter please read [Theoretical Concepts: Dark Matter (DM)](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Dark%20Matter%20(DM).md)

## Contents

Soon.

## 1. Getting Started

These instructions explain how to get a copy of the project to run it on your local machine for development and testing purposes.

### 1.1 Installation

The project can be either cloned or downloaded to your own device.<br/>The source code of the application is implemented in Python, with the requirement of the following open-source libraries (version numbers are up to date: 21.06.2020):

* Keras (2.3.1)
* TensorFlow (2.1.0)
* Numpy (1.18.1)
* scikit-learn (0.22.1)
* scikit-image (0.9.3)
* OpenCV-Python (4.0.1)
* imgaug (0.4.0)
* Matplotlib (3.2.1)
* Pillow (7.1.2)

The libraries can manually be installed from the anaconda prompt with the command ``` pip install 'library_name' ```. Make sure the correct working environment is activated.
If a module cannot be properly installed (installing tensorflow might sometimes be troublesome), doing it through the anaconda navigator is a good option.

## 2. Python files

In  this  section  the  core  of  the  project  is  dissected.   Every  employed method is explained;  from the origination of an image,  to the training of the model, every necessary step in the creation of the deep learning application is analyzed.

### 2.1 image_details.py

* **Function**: sets details of the simulated images that are created in image_simulation.py. The pixel intensity value of each element in the image can be defined.

For more information regarding the image simulation please read [Theoretical Concepts: Image Simulation](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Machine%20Learning%20(ML)/Image%20Simulation.md)

### 2.2 image_simulation.py

* **Function**: creates simulated images. Parameters such as number of images, elements, noise, glowing, etc. can be defined.
Images are saved to the saving path; arrays containing the predefined pixel intensity values are saved.<br/>
Testing and training images can be created. There is no need to create training and validating images on different folders. The model automatically shuffles all images and creates a validation split (with the defined size) when training.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Simulated_CCD_Image.png" width="400"/>
</p>

*Simulated 256X256 pixel CCD image containing glowing, hot pixels and pixel clusters. The pixel intensity values are given in ADCs.*
* **Caution**: it is important to be aware of the importance of the predefined pixel intensity values.
The way this model is implemented, image lables do not need to be provided. Image labels are directly obtained from the images.
In order to do this, image pixel intensity values are taken as reference to label different classes (please read [mask.py](https://github.com/aritzLizoain/Image-segmentation#maskpy) for more information).
Therefore elements with overlapping pixel intensity values will not be correctly labeled.

* **Requirements**: working directory path must contain [image_details.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/image_details.py) and [Cluster.pkl](https://github.com/aritzLizoain/Image-segmentation/blob/master/Cluster.pkl).

For more information regarding the image simulation please read [Theoretical Concepts: Image Simulation](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Machine%20Learning%20(ML)/Image%20Simulation.md)

### 2.3 load_dataset.py NEEDS TO BE UPDATED

* **Function**: receives the images and saves them as numpy arrays with shape (n_img, h, w, 3(rgb)), where n_img = # images, h = height, w = width. 

  * [get_weights](https://github.com/aritzLizoain/Image-segmentation/blob/0fc6f36abc9fcc63aee3c5129989fff54891147e/load_dataset.py#L52)
    is used to avoid issues with imbalanced datasets, where images are too biased towards one class. 
    In a case where >95% of the pixels are labeled as background, and <1% as clusters, the model can give a 95% accuracy prediction, but the predicted label will be all black, predicted as background.
    The weight of each class is obtained as the inverse of its presence percentage over all the training samples.
    The weights are then normalized to the number of classes.
    These weights are used by the model in the [weighted_categorical_crossentropy loss function](https://github.com/aritzLizoain/Image-segmentation/blob/2bd248e3c63bdad6823edbf883343b6f84f4536e/models.py#L29).

* **Caution**: make sure the path is correct. If it is not, it will not be able to load any data.

### 2.4 mask.py NEEDS TO BE UPDATED

* **Function**: it works with the images, creating masks, creating labels from masks and getting image statistics. 

  * [get_monochrome](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L59) converts the input image into a monochrome image.<br/>Input shape = (n_img, h, w, 3(rgb)) --> Output shape = (n_img, h, w, 1), where n_img = # images, h = height, w = width.
  * [get_class](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L66) defines the class of each pixel applying threshold values that can be defined.
  * [get_mask](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L89) creates masks from input images. It uses [get_monochrome](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L59) and [get_class](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L66).<br/>Input shape = (n_img, h, w, 3(rgb)) --> Output shape = (n_img, h, w, n_classes), where n_classes = # classes.
  * [get_max_in_mask](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L118) takes the position of the maximum  value, i.e., the class.<br/>Input shape = (n_img, h, w, n_classes) --> Output shape = (n_img, h, w, 1).
    Example of [get_mask](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L89) and [get_max_in_mask](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L118): a pixel is class = 2. Then n_classes = [0,0,1,0] and get_max_in_mask will return the value 2.
  * [mask_to_label](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L136) takes the mask and creates a label that can be visualized. It applies a defined color multiplier to each class.<br/>Input shape = (n_img, h, w, 1) --> Output shape = (n_img, h, w, 3(rgb)).
  * [statistics](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L168) shows the number of classes and the respective presence percentages on the dataset.
  * [get_percentages](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L182) simply returns the percentages of each class. This is used to calculate the weights for the loss function by [get_weights](https://github.com/aritzLizoain/Image-segmentation/blob/0fc6f36abc9fcc63aee3c5129989fff54891147e/load_dataset.py#L52).
  * [visualize_label](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L197) is used to visualize the created label.
  * [create_masks](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L222) takes the images as input and returns the masks (created by the previous functions). These masks are what the model uses to train and evaluate the model while training.
  * [create_labels](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L230) takes the images as input and returns the labels (created by the previous functions) that can be visualized. 
  * [create_labels_noStat_noPrint](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L246) is the same as [create_labels](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L230) but it does not print the information in the console. Done in order to avoid the repeated information shown by the console.
  * [output_to_label](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L258) takes the masks predicted by the model and converts them into labels that can be visualized. IMPORTANT: the model does not work with the labels that are visualized and does not predict the labels that are visualized .The model works and predicts masks with shape (n_img, h, w, n_classes).

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Image_label_representation.png" width="600"/>
</p>

*Image label example, where each pixel is classified as a class and its corresponding depth channel takes value 1. Adapted Digital Image. Jordan, J. "An overview of semantic image segmentation". (2018). [Link](https://www.jeremyjordan.me/semantic-segmentation).*

* **Caution**: it is important to be aware of a possible issue regarding the color of the elements.
The way this model is implemented, image lables do not need to be provided. Image labels are directly obtained from the images.
In order to do this, image pixel values, i.e., colors, are taken as reference to label different classes.
Therefore a color change of an object in the image can cause a wrong label creation if this has not been correctly specified in [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py).<br/>
Labels can perfectly be created using a labeling software. However, for the purpose of this project, automatic pixel-wise labeling is a practical solution. Remember that, in case of using your own labels, image and label names must match.

For more information regarding the labeling process please read [Theoretical Concepts: Image Labeling](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Machine%20Learning%20(ML)/Image%20Labeling.md)
   

### 2.5 augmentation.py NEEDS TO BE UPDATED

* **Function**: applies data augmentation techniques to both images and corresponding labels. Due to the type of images working with, non-geometric augmentation can lead to wrong labeling.
Therefore only geometric augmentation is applied: flip, crop, pad, scale, translate and rotate.
Please read the [imgaug documentation](https://imgaug.readthedocs.io/en/latest/index.html) for more information on augmentation techniques.
The original image and label, and augmented ones, are visualized.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Augmentation.png" width="400"/>
</p>

*Image and label augmentation example. The applied transformations are translation and scaling.*

For more information regarding data augmentation please read [Theoretical Concepts: Data Augmentation](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Data%20Augmentation.md)

### 2.6 models.py NEEDS TO BE UPDATED

The  U-Net  structure  is  implemented  in  the  model.py  file.   Every  layer composing  the  CNN  and  each  hyper-parameter  is  specified  in  it.   In addition, the weighted categorical crossentropy loss function is defined.

* **Function**: defines the model architecture and layer features. The model has UNet architecture. The code is already prepared to add layers into the model. Layers can be removed too. Additionally, pretrained weights from an already trained model can be used.

  * [weighted_categorical_crossentropy loss function](https://github.com/aritzLizoain/Image-segmentation/blob/2bd248e3c63bdad6823edbf883343b6f84f4536e/models.py#L29)
    is used to calculate the categorical crossentropy loss of the model with the addition of taking into account the weight of each class.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Unet.png" width="800"/>
</p>

*Implemented U-Net architecture. Adapted from Ronneberger, O. et al. "Convolutional Networks for Biomedical Image Segmentation". (2015). [Link](https://arxiv.org/pdf/1505.04597.pdf).*

For more information regarding ML (CNNs, layers, image segmentation) please read [Theoretical Concepts: Machine Learning](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Machine%20Learning.md)

For more information regarding the implementation of the network (employed layers, activation functions, loss function, optimizer) please read [Theoretical Concepts: Network Implementation](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Network%20Implementation.md)

   
### 2.7 train.py NEEDS TO BE UPDATED

* **Function**: training the model. This is the MAIN CODE. Process:
  * Loads the images.
  * Creates the labels for visualization.
  * Applies augmentation on both images and labels.
  * Creates the masks for training the model.
  * Trains the model with the defined hyperparameters and callbacks. For more information regarding callbacks please read the [keras callbacks documentation](https://keras.io/api/callbacks/).
  * Plots and saves the accuracy and loss over the training process.
  * Predicts train and test image masks.
  * Converts predicted masks into labels that can be visualized.
  * Plots original images, labels, and predicted label comparisons.
  * Evaluates the model on the test set.
  * Gives a classification report that analyzes the performance of each class. For more information regarding the classification reports please read the [scikit-learn classification report documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).

  All figures, accuracy and loss data throughout the training process, and trained model are saved in the defined paths.
  ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/console.gif "Training console display")
  
  *Example of   the console display while training*

* **Caution**: make sure all paths are correct. If they are not, it will not be able to load any data.<br/>Depending on the used device, training a model can be rather slow (>10'/epoch), particularly when large datasets and number of epochs are being used.

* **Requirements**: working directory path must contain [load_dataset.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/load_dataset.py), [models.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/models.py), [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py) and [augmentation.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/augmentation.py).

For more information regarding the model training process please read [Theoretical Concepts: Model Training](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Model%20Training.md)

### 2.8 load_model.py NEEDS TO BE UPDATED

* **Function**: loads an already trained model. This loaded model is used to make predictions on test images, evaluate the model and give a classification report (same as the last section of [train.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/train.py#L147)).<br/>
All figures are saved in the defined path.

* **Requirements**: working directory path must contain [load_dataset.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/load_dataset.py), [models.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/models.py) and [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py).

## 3. Implementation summary

The Python application consists on the 8 files previously explained. Only the last two ([train.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/train.py) and [load_model.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/load_model.py)) are executed.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Summary.png" width="1000"/>
</p>

*Python implementation summary.*

## 4. Results NEEDS TO BE UPDATED

An imbalanced dataset entails further problems. A good solution to this issue has been creating balanced images, with approximately the same percentage of presence of each class. The classes have not been mixed in order to avoid confusion to the model when labeling the images.
Here is an example of an image used for training the model: ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/training_example.png "Training image example")

60 images have been used ([Train images](https://github.com/aritzLizoain/Image-segmentation/tree/master/Images/Train)). The model has been trained for 100 epochs with the following hyperparameters: 
* split = 0.2
* weights = [0.514, 0.840, 0.983, 1.663] <-- given by [get_weights](https://github.com/aritzLizoain/Image-segmentation/blob/0fc6f36abc9fcc63aee3c5129989fff54891147e/load_dataset.py#L52)
* activation = 'elu' <br/>
  For more information regarding activation functions please read the [keras layer activation functions documentation](https://keras.io/api/layers/activations/).
* dropout = 0.18
* loss = weighted_categorical_crossentropy(weights)
* optimizer = 'adadelta' <br/>
  Adadelta optimization is a stochastic gradient descent method that is based on adaptive learning rate. There is no need to manually set any default learning rate. For more information regarding [adadelta](https://keras.io/api/optimizers/adadelta/) and other optimizers please read the [keras optimizers documentation](https://keras.io/api/optimizers/).<br/>

The rest of parameters have been left as default. Please note that these parameters work well for this particular dataset, but do not assure reliable results for all kind of datasets.

<pre>
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/accuracy_100epochs.png" width="400"/>           <img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/loss_100epochs.png" width="388"/> 
</pre>

Prediction on a training image: ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/training_prediction.png "Training image prediction")

Prediction on test images: ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/test4.png "Test image 4 prediction")
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/test6.png "Test image 6 prediction")

The accuracy of the model on the test set is: 99.67%<br/>
The loss of the model on the test set is: 0.166<br/>
Classification report:
Class        | Precision| Recall| F1-score| Support
---          | ---      | ---   | ---     | ---
Background   | 1.00     | 1.00  | 1.00    | 510811
Glowing      | 0.98     | 0.93  | 0.96    | 12030
Hot pixel    | 1.00     | 0.95  | 0.97    | 674
Cluster      | 0.25     | 0.60  | 0.35    | 773
**Accuracy**     |          |       | 1.00    | 524288
**Macro avg.**   | 0.81     | 0.88  | 0.83    | 524288
**Weighted avg.**| 1.00     | 1.00  | 1.00    | 524288

_Precision_: the percentage of correctly classified pixels among all pixels classified as the given class.<br/> 
_Recall_: the percentage of correctly classified pixels among all pixels that truly are of the given class.<br/> 
_F1-score_: the harmonic mean between precision & recall. Useful to analyze the performance on inbalanced sets.<br/>
Best score is 1.00 and worst score is 0.00.<br/>
_Support_: the number of pixels of the given class in the dataset.

## 5. Future steps NEEDS TO BE UPDATED

* Particle identification.

## 6. Contributing

Feel free to submit pull requests.

Please read [CONTRIBUTING.md](https://github.com/aritzLizoain/Image-segmentation/blob/master/CONTRIBUTING.md) for details on the code of conduct, and the process for submitting pull requests.

If you use this code in a publicly available project, please post an issue or create a pull request and your project link will be added here.

## 7. Versioning NEEDS TO BE UPDATED

* First release: [1.0 CNN on simulated images](https://github.com/aritzLizoain/Image-segmentation/releases/tag/v1.0).

Azalpena

* Second release: [2.0 CNN Image Segmentation](https://github.com/aritzLizoain/Image-segmentation/releases/tag/v2.0).

Soon

## 8. Acknowledgments

I express my sincere gratitude to my director, Rocío Vilar Cortabitarte, and co-director, Alicia Calderón Tazón, for providing their expertise and guidance throughout the course of this project. I would also like to thank the rest of my advisors, Agustín Lantero Barreda and Núria Castelló-Mor, who contributed so thoroughly through their assistance and dedicated involvement.

## 9. Copyright

Copyright 2020, Aritz Lizoain, All rights reserved.
