# CNN Image Segmentation
## Application of deep learning techniques to images collected with Charge Coupled Devices to search for Dark Matter


The Standard Model of particle physics, while being able to make accurate predictions, has been proved to fail to explain various phenomena, such as astronomical dark matter observations.
In this work, a machine learning application has been implemented with the goal of studying dark matter candidates.
Images from Charge Coupled Devices (CCDs) in different experiments DAMIC/DAMIC-M  located underground are used to test different deep learning algorithms.
A U-Net model is trained with Python's open-source library Keras. The model performs multi-class image segmentation in order to detect dark matter particle signals among background noise.

:information_source: For more information regarding dark matter please read [Theoretical Concepts: Dark Matter (DM)](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Dark%20Matter%20(DM).md)

## Contents

Soon.

1. :computer: [Getting Started](https://github.com/aritzLizoain/Image-segmentation#1-computer-getting-started) 
  * 1.1 [Installation](https://github.com/aritzLizoain/Image-segmentation#11-installation)

2. :snake: :page_facing_up: [Python files](https://github.com/aritzLizoain/Image-segmentation#2-snake-page_facing_up-python-files) 
  * 2.1 [image_details.py](https://github.com/aritzLizoain/Image-segmentation#21-image_detailspy)
  * 2.2 [image_simulation.py](https://github.com/aritzLizoain/Image-segmentation#22-image_simulationpy)
  * 2.3 [load_dataset.py](https://github.com/aritzLizoain/Image-segmentation#23-load_datasetpy)
  * 2.4 [mask.py](https://github.com/aritzLizoain/Image-segmentation#24-maskpy)
  * 2.5 [augmentation.py](https://github.com/aritzLizoain/Image-segmentation#25-augmentationpy)
  * 2.6 [models.py](https://github.com/aritzLizoain/Image-segmentation#26-modelspy)
  * 2.7 [train.py](https://github.com/aritzLizoain/Image-segmentation#27-trainpy)
  * 2.8 [load_model](https://github.com/aritzLizoain/Image-segmentation#28-load_modelpy-needs-to-be-updated)

3. :rocket: [Implementation summary](https://github.com/aritzLizoain/Image-segmentation#3-rocket-implementation-summary) 

4. :dart: [Results](https://github.com/aritzLizoain/Image-segmentation#4-dart-results-needs-to-be-updated) 

5. :thought_balloon: :soon: [Future steps](https://github.com/aritzLizoain/Image-segmentation#5-thought_balloon-soon-future-steps-needs-to-be-updated)  

6. 🤝 [Contributing](https://github.com/aritzLizoain/Image-segmentation#6--contributing)  

7. :egg: :hatching_chick: :hatched_chick: [Versioning](https://github.com/aritzLizoain/Image-segmentation#7-egg-hatching_chick-hatched_chick-versioning-needs-to-be-updated)  

8. :family: [Acknowledgements](https://github.com/aritzLizoain/Image-segmentation#8-family-acknowledgements)

9. :copyright: [Copyright](https://github.com/aritzLizoain/Image-segmentation#9-copyright-copyright)  

 

## 1. :computer: Getting Started

These instructions explain how to get a copy of the project to run it on your local machine for development and testing purposes.

### 1.1 Installation

The project can be either cloned or downloaded to your own device. The source code of the application is implemented in Python, with the requirement of the following open-source libraries (version numbers are up to date: 21.06.2020):

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

## 2. :snake: :page_facing_up: Python files

In  this  section  the  core  of  the  project  is  dissected.   Every  employed method is explained;  from the origination of an image,  to the training of the model, every necessary step in the creation of the deep learning application is analyzed.

### 2.1 image_details.py

:chart_with_upwards_trend: **Function**: sets details of the simulated images that are created in [image_simulation.py](https://github.com/aritzLizoain/Image-segmentation#22-image_simulationpy). The pixel intensity value of each element in the image can be defined.

:information_source: For more information regarding the image simulation please read [Theoretical Concepts: Image Simulation](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Image%20Simulation.md)

### 2.2 image_simulation.py

:chart_with_upwards_trend: **Function**: creates simulated images. Parameters such as number of images, elements, noise, glowing, etc. can be defined.
Images are saved to the saving path; arrays containing the predefined pixel intensity values are saved.<br/>
Testing and training images can be created. There is no need to create a validation dataset. The model automatically shuffles all images and creates a validation split in the training process.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Simulated_CCD_Image.png" width="400"/>
</p>

*Simulated 256X256 pixel CCD image containing glowing, hot pixels and pixel clusters. The pixel intensity values are given in ADCs.*

:warning: **Caution**: it is important to be aware of the importance of the predefined pixel intensity values.
The way this model is implemented, image lables do not need to be provided. Image labels are directly obtained from the images.
In order to do this, image pixel intensity values are taken as reference to label different classes (please read [mask.py](https://github.com/aritzLizoain/Image-segmentation#24-maskpy) for more information).
Therefore elements with overlapping pixel intensity values will not be correctly labeled.

:cop: **Requirements**: working directory path must contain [image_details.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/image_details.py) and [Cluster.pkl](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/Cluster.pkl).

:information_source: For more information regarding the image simulation please read [Theoretical Concepts: Image Simulation](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Image%20Simulation.md)

### 2.3 load_dataset.py

:chart_with_upwards_trend: **Function**: calculates the weights for the loss function and processes FITS files (DAMIC images). Originaly created to load the datasets as PNG files into arrays (unused in version 2.0).
  
  * [load_images](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/load_dataset.py#L27)(commented) is no longer used in version 2.0. It was used in version 1.0 for loading the PNG images.

  * [get_weights](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/load_dataset.py#L59)
    is used to avoid issues with imbalanced datasets, where images are too biased towards one class. 
    In a case where >95% of the pixels are labeled as background, and <1% as clusters, the model can give a 95% accuracy prediction, but all pixels might be predicted as background, giving a meaningless output.
    Please read [Theoretical Concepts: Network Implementation (Loss Function)](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Network%20Implementation.md#loss-function) for more information.
    The weight of each class is obtained as the inverse of its frequency in the training samples.
    The weights are then normalized to the number of classes.
    These weights are used by the model in the [weighted_categorical_crossentropy loss function](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/models.py#L31).

  * [process_fits](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/load_dataset.py#L73) is used to process the DAMIC images, which are given in FITS (Flexible Image Transport System) file format.
    These files are read and saved as arrays that contain the collected charge by each CCD pixel. Since 256X256 pixel images are used for training the model, the DAMIC image is divided into sections of the same size, so they can be individually passed through the trained model, obtaining their respective predicted labels.
    The possibility to normalize, cut and resize the image is given.

  * [images_small2big](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/load_dataset.py#L183) is used to reconstruct the predictions of all sections into a full segmentation map.

  * [check_one_object](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/load_dataset.py#L201) is used to analyze the final output. It looks for a chosen category (i.e. 'Clusters') section by section. It returns a modified predicted label; it only shows background and the pixels classified as a chosen class.

### 2.4 mask.py

:chart_with_upwards_trend: **Function**: creates the labels in an automated way, visualizes the output labels and can analyze and modify the images.

  * [get_monochrome](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L52) converts the input image into a monochrome image. Input shape = (number of images, height, width, 3(RGB)) --> Output shape = (number of images, height, width, 1).
  * [get_class](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L62) classifies each pixel as a certain class depending on its intensity value.
  * [get_mask](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L87) creates labels from input images. Input shape = (number of images, height, width, 3(RGB)) --> Output shape = (number of images, height, width, number of classes), where number of classes = 4 in this project.
  * [get_max_in_mask](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L116) takes the position of the maximum  value, i.e., the class. Input shape = (number of images, height, width, number of classes) --> Output shape = (number of images, height, width, 1).
  * [mask_to_label](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L126) creates the segmentatin map that can be visualized. It applies a color multiplier to each class. Input shape = (number of images, height, width, 1) --> Output shape = (number of images, height, width, 3(RGB)).
  * [mask_to_label_one_object](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L167) creates the segmentatin map to visualize a chosen class. It is used by [check_one_object](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/load_dataset.py#L201). It applies a color multiplier to that one class. Input shape = (number of images, height, width, 1) --> Output shape = (number of images, height, width, 3(RGB)).
  * [statistics](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L208) shows the number of classes and their frequency on the dataset.
  * [get_percentages](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L222) simply returns the percentages of each class. This is used to calculate the weights for the loss function by [get_weights](https://github.com/aritzLizoain/Image-segmentation/blob/0fc6f36abc9fcc63aee3c5129989fff54891147e/load_dataset.py#L52).
  * [visualize_label](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L236) is used to visualize the created label.
  * [create_masks](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L261) takes the images as input and returns the labels. These labels are used by the model to train and evaluate the model while training.
  * [create_labels](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L269) takes the images as input and returns the segmentation maps that can be visualized. 
  * [create_labels_noStat_noPrint](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L284) is the same as [create_labels](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py#L230) but it does not print the information in the console. Created in order to avoid the repeated information shown by the console.
  * [output_to_label](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L294) takes the model outputs and converts them into labels that can be visualized. 
  * [output_to_label_one_object](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L305) takes the model outputs and converts them into labels where a chosen class is visualized. 
  * [rgb2random](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L315) randomly changes the RGB colors of the images (unused).
  * [rgb2gray](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L336) converts the images to grayscale (unused).
  * [contrast](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L355) augments the contrast of the image (unused). 
  * [percentage_result](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/mask.py#L378) gives the percentage of each class in an output label.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Image_label_representation.png" width="600"/>
</p>

*Image label example, where each pixel is classified as a class and its corresponding depth channel takes value 1. Adapted Digital Image. Jordan, J. "An overview of semantic image segmentation". (2018). [Link](https://www.jeremyjordan.me/semantic-segmentation).*

:warning: **Caution**: it is important to be aware of the importance of the predefined pixel intensity values in [image_details.py](https://github.com/aritzLizoain/Image-segmentation#21-image_detailspy). The way this model is implemented, image lables do not need to be provided. Image labels are directly obtained from the images. In order to do this, image pixel intensity values are taken as reference to label different classes. Therefore elements with overlapping pixel intensity values will not be correctly labeled.<br/><br/>Labels can perfectly be created using a labeling software. However, for the purpose of this project, automatic pixel-wise labeling is a practical solution. Remember that, in case of using your own labels, image and label names must match.

:information_source: For more information regarding the labeling process please read [Theoretical Concepts: Image Labeling](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Image%20Labeling.md)
   

### 2.5 augmentation.py

:chart_with_upwards_trend: **Function**: applies data augmentation techniques to both images and corresponding labels. Please read the [imgaug documentation](https://imgaug.readthedocs.io/en/latest/index.html) for more information on augmentation techniques. This is an optional step; it is applied when only a few training samples are available, or when the desired property is not present in the dataset.

  * [augmentation_sequence_Color](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/augmentation.py#L20) and [augmentation_Color](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/augmentation.py#L29) apply color dropout, rotation, flipping.
  * [augmentation_sequence_Invert](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/augmentation.py#L77) and [augmentation_Invert](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/augmentation.py#L86) apply color channel inversion, dropout, logContrast, hue, gammaContrast.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Augmentation.png" width="400"/>
</p>

*Image and label augmentation example. The applied transformations are translation and scaling.*

:information_source: For more information regarding data augmentation please read [Theoretical Concepts: Data Augmentation](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Data%20Augmentation.md)

### 2.6 models.py

:chart_with_upwards_trend: **Function**: defines the model architecture and layer features. The model has U-Net architecture. The code is already prepared to add or remove layers in the model. Additionally, pretrained weights from an already trained model can be used.

  * [weighted_categorical_crossentropy loss function](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/models.py#L26)
    is used to calculate the categorical crossentropy loss of the model with the modification of taking into account the weight of each class.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Unet.png" width="800"/>
</p>

*Implemented U-Net architecture. Adapted from Ronneberger, O. et al. "Convolutional Networks for Biomedical Image Segmentation". (2015). [Link](https://arxiv.org/pdf/1505.04597.pdf).*

:information_source: For more information regarding ML (CNNs, layers, image segmentation) please read [Theoretical Concepts: Machine Learning](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Machine%20Learning.md)

:information_source: For more information regarding the implementation of the network (employed layers, activation functions, loss function, optimizer) please read [Theoretical Concepts: Network Implementation](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Network%20Implementation.md)

   
### 2.7 train.py

:chart_with_upwards_trend: **Function**: trains a model with the defined parameters. Process:
  * Loads the datasets.
  * Creates the labels.
  * (Optional) Applies augmentation on both images and labels.
  * Trains the model with the defined hyperparameters and callbacks. For more information regarding callbacks please read the [keras callbacks documentation](https://keras.io/api/callbacks/).
  * Plots and saves the accuracy and loss over the training process.
  * Gives the predicted outputs of training and testing images.
  * Evaluates the model on the test set.
  * Gives a classification report that analyzes the performance of each class. For more information regarding the classification reports please read the [scikit-learn classification report documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).

  The trained model and all the figures are saved in the defined paths.

  ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/console.gif "Training console display")
  
  *Example of   the console display while training*

:warning: **Caution**: Depending on the used device, training a model can be rather slow, particularly when large datasets and number of epochs are being used. If the model name is not changed, the model will be overwritten.

:cop: **Requirements**: working directory path must contain [models.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/models.py), [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py) and [augmentation.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/augmentation.py)(if used).

:information_source: For more information regarding the model training process please read [Theoretical Concepts: Model Training](https://github.com/aritzLizoain/Image-segmentation/blob/master/Theoretical%20Concepts/Model%20Training.md)

### 2.8 load_model.py

:chart_with_upwards_trend: **Function**: loads an already trained model and makes predictions on the FITS file (the DAMIC CCD image). Process:
  * Loads the trained model.
  * Loads and processes the FITS file.
  * Creates small sections from the DAMIC image.
  * Passes each section individually through the model.
  * Reconstructs all sections into a prediction label.
  * Looks for a chosen category (i.e. 'Clusters') section by section.

:cop: **Requirements**: working directory path must contain [load_dataset.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/load_dataset.py), [models.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/models.py) and [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py).

## 3. :rocket: Implementation summary

The Python application consists on the 8 files previously explained. Only the last two ([train.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/train.py) and [load_model.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/Code/load_model.py)) are executed.

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Summary.png" width="1000"/>
</p>

*Python implementation summary.*

## 4. :dart: Results NEEDS TO BE UPDATED

An imbalanced dataset entails further problems. A good solution to this issue is creating balanced images, with approximately the same percentage of presence of each class. The classes are not mixed in order to avoid confusion to the model when labeling the images.
Here is an example of an image used for training the model: 

<p align="center">
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Traing/Training_image_example.png" width="400"/>
</p>

*Simulated 256×256 pixel image with a similar number of pixels belonging to each class.*



only 60%of  the  images  contained  glowing,  and  it  did  not  always  start  from  thefirst pixel.  This way, the model did not learn that all predictions shouldhave a glowing column, nor where should it be.


For this project 200 training and 42 test images were created. As previously explained, each image contained glowing on the left side, and hot pixels and clusters randomly placed on the right (see \ref{fig:3}). From the training set 42 samples were taken for validation. The network was set with an 18$\%$ dropout. Small variations of this value (10-25$\%$ is the common dropout range) did not significantly alter the final result. Taking the original U-Net model as a reference, a batch size of 1 sample was set. The training was defined for 100 epochs. 

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

The loss (see Figure 19Left) and accuracy (see Figure 19Right) of the training and evaluationset verified that the model did not suffer from overfitting.

<pre>
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Accuracy.png" width="400"/>           <img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Loss.png" width="400"/> 
</pre>

* CHANGE CAPTION Example of a correct training of a model that reaches its optimum state at epoch 40.* <br/> LEFT: *Training and validation accuracy.* <br/> *RIGHT: Training and validation loss.*


Prediction on a training image: ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/training_prediction.png "Training image prediction")

Prediction on test images: ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/test4.png "Test image 4 prediction")
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/test6.png "Test image 6 prediction")

The  model  performed  correctly  on  the  test  dataset,  segmenting  everyobject and reaching a 99.2% accuracy.  The classification report showedhow efficiently each class performed (see Table 1).

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

The model also gave a seemingly correct prediction of a DAMIC image(T=140K). Due to the small size of the objects, these could not be seenwhen the whole image was displayed.  If the 256×256 sections were indi-vidually observed instead, the segmented clusters could be analyzed (seeFigure 20).

FIGURE SECTION

## 5. :thought_balloon: :soon: Future steps NEEDS TO BE UPDATED

* Particle identification.

## 6. 🤝 Contributing

Feel free to submit pull requests.

Please read [CONTRIBUTING.md](https://github.com/aritzLizoain/Image-segmentation/blob/master/CONTRIBUTING.md) for details on the code of conduct, and the process for submitting pull requests.

If you use this code in a publicly available project, please post an issue or create a pull request and your project link will be added here.

## 7. :egg: :hatching_chick: :hatched_chick: Versioning

* First release: [1.0 CNN on simulated images](https://github.com/aritzLizoain/Image-segmentation/releases/tag/v1.0).

Only for simulated images. Not for real images.

* Second release: [2.0 CNN Image Segmentation](https://github.com/aritzLizoain/Image-segmentation/releases/tag/v2.0).

For both simulated and real images.

## 8. :family: Acknowledgements

I express my sincere gratitude to my director, Rocío Vilar Cortabitarte, and co-director, Alicia Calderón Tazón, for providing their expertise and guidance throughout the course of this project. I would also like to thank the rest of my advisors, Agustín Lantero Barreda and Núria Castelló-Mor, who contributed so thoroughly through their assistance and dedicated involvement.

## 9. :copyright: Copyright

Copyright 2020, Aritz Lizoain, All rights reserved.

<a href="#top">:top:</a>