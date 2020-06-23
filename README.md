# CNN UNet Image Segmentation

The goal of the project is to study machine learning techniques on detector images from Beyond Standard Model searches.
The studied images represent energy deposits on CCDs.
The project performs CNN UNet multiclass image segmentation.
The model is trained using simulated detector images and then tested with real detector images.

Example of a real detector image: ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/DAMIC_image.png "Detector image example")

## Getting Started

These instructions explain how to get a copy of the project to run it on your local machine for development and testing purposes.

### Installation

The project can be either cloned or downloaded to your own device.<br/>The following requirements need to be installed (version numbers are up to date: 21.06.2020):

* keras (2.3.1)
* tensorflow (2.1.0)
* numpy (1.18.1)
* scikit-learn (0.22.1)
* scikit-image (0.9.3)
* opencv-python (4.0.1)
* imgaug (0.4.0)
* matplotlib (3.2.1)
* pillow (7.1.2)

You can manually install the libraries from the anaconda prompt with the command ``` pip install 'library_name' ```. Make sure the correct working environment is activated.
If a module cannot be properly installed (installing tensorflow might sometimes be troublesome), doing it through the anaconda navigator is a good option.

## Python files

### image_details.py

* **Function**: sets details of the simulated images that are created in image_simulation.py.

### image_simulation.py

* **Function**: creates simulated images. Parameters such as number of elements, noise, glowing, images, etc. can be defined.
Images are saved to the saving path.<br/>
There is no need to create training and validating images on different folders. The model automatically shuffles all images and creates a validation split (with the defined size) when training.

* **Caution**: it is important to be aware of a possible issue regarding the color of the elements.
The way this model is implemented, image lables do not need to be provided. Image labels are directly obtained from the images.
In order to do this, image pixel values, i.e., colors, are taken as reference to label different classes (please read [mask.py information](https://github.com/aritzLizoain/Image-segmentation#maskpy) for more information).
Therefore a color change of an object in the image can cause a wrong label creation if this has not been correctly specified in [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py).

* **Requirements**: working directory path must contain [image_details.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/image_details.py) and [Cluster.pkl](https://github.com/aritzLizoain/Image-segmentation/blob/master/Cluster.pkl).

### load_dataset.py

* **Function**: receives the images and saves them as numpy arrays with shape (n_img, h, w, 3(rgb)), where n_img = # images, h = height, w = width. 

  * [get_weights](https://github.com/aritzLizoain/Image-segmentation/blob/0fc6f36abc9fcc63aee3c5129989fff54891147e/load_dataset.py#L52)
    is used to avoid issues with imbalanced datasets, where images are too biased towards one class. 
    In this case >95% of the pixels are labeled as background, and only <1% as clusters. 
    This way the model can give a 95% accuracy prediction, but the predicted label will be all black, predicted as background.
    The weight of each class is obtained as the inverse of its presence percentage over all the training samples.
    Then the weights are normalized to the number of classes.
    These weights are used by the model in the [weighted_categorical_crossentropy loss function](https://github.com/aritzLizoain/Image-segmentation/blob/2bd248e3c63bdad6823edbf883343b6f84f4536e/models.py#L29).

* **Caution**: make sure the path is correct. If it is not, it will not be able to load any data.

### models.py

* **Function**: defines the model architecture and layer features. The model has UNet architecture. The code is already prepared to add layers into the model. Layers can be removed too. Additionally, pretrained weights can be used.

  * [weighted_categorical_crossentropy loss function](https://github.com/aritzLizoain/Image-segmentation/blob/2bd248e3c63bdad6823edbf883343b6f84f4536e/models.py#L29)
    is used to calculate the categorical crossentropy loss of the model with the addition of taking into account the weight of each class.

#### More information regarding CNNs, UNet, layers, hyperparameter optimization, etc.
* https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
* https://medium.com/intuitive-deep-learning/intuitive-deep-learning-part-2-cnns-for-computer-vision-24992d050a27
* https://medium.com/@jcrispis56/introducci%C3%B3n-al-deep-learning-parte-2-redes-neuronales-convolucionales-f743266d22a0 (in Spanish)
* https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
* https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca
* https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47 (UNet)

  UNet architecture:
  ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Models/Architecture%201.png "UNet architecture 1")

  ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Models/Architecture%202.png "UNet architecture 2")
* Layers summary:
  - Convolution: uses parameter sharing and applies the same smaller set of parameters spatially across the image. Filter size corresponds to how many input features in the width and height dimensions one neuron takes in. Stride helps overlapping regions, it defines how many pixels we want to move when applying the neuron again. Depth defines how many different outputs/classes do we have. Padding adds a border of 0s.
  - Upsampling convolution: learns parameters through back propagation to convert a low resolution image to a high resolution image. It needs to concatenate with the corresponding downsampling layer.
  - Max Pool: takes the maximum of the numbers it looks at. Applies to each individual depth channel separately; leaves depth dimension unchanged. It is only defined by filter size and stride, reducing the spatial size by taking the maximum of the numbers within its filter.
  - Dropout: during training, some number of layer outputs are randomly ignored or “dropped out. It is applied in order to avoid overfitting.
  - Fully-Connected (FC): every neuron in the next layer takes as input every neuron in the previous layer's output. Usually used at the end of the CNNs. We can flatten the neurons into a one-dimensional array of features.
  - Softmax: transforms the output of the previous layer into probability distributions, which is the last layer.
   
### mask.py

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

  For more information regarding the labeling process please read https://www.jeremyjordan.me/semantic-segmentation/
  ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/labels.png "labeling example 1")

  ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/labels2.png "labeling example 2")
* **Caution**: it is important to be aware of a possible issue regarding the color of the elements.
The way this model is implemented, image lables do not need to be provided. Image labels are directly obtained from the images.
In order to do this, image pixel values, i.e., colors, are taken as reference to label different classes.
Therefore a color change of an object in the image can cause a wrong label creation if this has not been correctly specified in [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py).<br/>
Labels can perfectly be created using a labeling software. However, for the purpose of this project, automatic pixel-wise labeling is a practical solution. Remember that, in case of using your own labels, image and label names must match.
    
### augmentation.py

* **Function**: applies data augmentation techniques to both images and corresponding labels. Due to the type of images working with, non-geometric augmentation can lead to wrong labeling.
Therefore only geometric augmentation is applied: flip, crop, pad, scale, translate and rotate.
Please read the [imgaug documentation](https://imgaug.readthedocs.io/en/latest/index.html) for more information on augmentation techniques.
The original image and label, and augmented ones, are visualized.

  Augmented image and label example:
  ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Augmentation_example.png "Augmentation example")

### train.py 

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

  All figures, accuracy and loss data throughout the training process, and trained model are saved in the defined paths.<br/>Example of   the console display while training:
  ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/console.gif "Training console display")

* **Caution**: make sure all paths are correct. If they are not, it will not be able to load any data.<br/>Depending on the used device, training a model can be rather slow (>10'/epoch), particularly when large datasets and number of epochs are being used.

* **Requirements**: working directory path must contain [load_dataset.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/load_dataset.py), [models.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/models.py), [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py) and [augmentation.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/augmentation.py).

### load_model.py

* **Function**: loads an already trained model. This loaded model is used to make predictions on test images, evaluate the model and give a classification report (same as the last section of [train.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/train.py#L147)).<br/>
All figures are saved in the defined path.

* **Requirements**: working directory path must contain [load_dataset.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/load_dataset.py), [models.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/models.py) and [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py).

## Results

The model has been trained with 200 simulated images during 1 epoch, obtaining a final accuracy of >99% on the test set, which containes other simulated images. It is able to predict correctly 3 classes: background, glowing and hot pixels. It fails to predict any cluster. Prediction on a test image:
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Test_1.png "Test image prediction 1")
Classification report of the test set:
Class        | Precision| Recall| F1-score| Support
---          | ---      | ---   | ---     | ---
Background   | 1.00     | 1.00  | 1.00    | 60793
Glowing      | 1.00     | 1.00  | 1.00    | 4352
Hot pixel    | 1.00     | 0.95  | 0.97    | 291
Cluster      | 0.00     | 0.00  | 0.00    | 100
**Accuracy**     |          |       | 1.00    | 65536
**Macro avg.**   | 0.75     | 0.74  | 0.74    | 65536
**Weighted avg.**| 1.00     | 1.00  | 1.00    | 65536

_Precision_: the percentage of correctly classified pixels among all pixels classified as the given class.<br/> 
_Recall_: the percentage of correctly classified pixels among all pixels that truly are of the given class.<br/> 
_F1-score_: the harmonic mean between precision & recall. Useful to analyze the performance on inbalanced sets.<br/>
Best score is 1.00 and worst score is 0.00.<br/>
_Support_: the number of pixels of the given class in the dataset.

## Future steps

* Being able to detect clusters.
* Testing the trained model on real detector images.

## Contributing

Feel free to submit pull requests.

Please read [CONTRIBUTING.md](https://github.com/aritzLizoain/Image-segmentation/blob/master/CONTRIBUTING.md) for details on the code of conduct, and the process for submitting pull requests.

If you use this code in a publicly available project, please post an issue or create a pull request and your project link will be added here.

## Versioning

Please check on [releases](https://github.com/aritzLizoain/Image-segmentation/releases) to find previous versions of the project.

## Acknowledgments

* Agustín Lantero for the [image_detais.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/image_details.py) and [image_simulation.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/image_simulation.py) codes.
* Rocío Vilar, Alicia Calderón and Nuria Castello-Mor for the help, advice and support. 

## Copyright

Copyright 2020, Aritz Lizoain, All rights reserved.
