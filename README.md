# CNN UNet Image Segmentation

The **goal** of the project is to study machine learning techniques on detector images from Beyond Standard Model searches.
The studied images represent energy deposits on CCDs.
The project performs CNN UNet multiclass image segmentation.
The model is trained using simulated detector images and then tested with real detector images.

Example of a real detector image: ![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/DAMIC_image.png "Detector image example")

## Getting Started

These instructions explain how to get a copy of the project, run it on your local machine for development and testing purposes.

### Installation

The project can be either cloned or downloaded to your own device.

The following requirements need to be installed (version numbers are up to date: 21.06.2020):

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

* **Function**: set details of the simulated images that are created in image_simulation.py.

* **Author**: Agustín Lantero.

### image_simulation.py

* **Function**: create simulated images. Parameters such as number of elements, noise, glowing, images, etc. can be defined.
Images are saved to the saving path.

* **Caution**: it is important to be aware of a possible issue regarding the color of the elements.
The way this model is implemented, image lables do not need to be provided. Image labels are directly obtained from the images.
In order to do this, image pixel values, i.e., colors, are taken as reference to label different classes (read [mask.py information](https://github.com/aritzLizoain/Image-segmentation#maskpy) for more information).
Therefore a color change of an object in the image can cause a wrong label creation if this has not been correctly specified in [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py).

* **Requirements**: working directory path must contain image_details.py and Cluster.pkl.

* **Author**: Agustín Lantero.

### load_dataset.py

* **Function**: takes images and saves them as numpy arrays. 

  * [get_weights](https://github.com/aritzLizoain/Image-segmentation/blob/0fc6f36abc9fcc63aee3c5129989fff54891147e/load_dataset.py#L52)
    function is used to avoid issues with imbalanced datasets, where images are too biased towards one class. 
    In this case >95% of the pixels are labeled as background, and only <1% as clusters. 
    This way the model can give a 95% accuracy prediction, but the predicted label will be all black, predicted as background.
    The weight of each class is obtained as the inverse of its presence percentage over all the training samples.
    Then the weights are normalized to the number of classes.
    These weights are used by the model in the [weighted_categorical_crossentropy loss function](https://github.com/aritzLizoain/Image-segmentation/blob/2bd248e3c63bdad6823edbf883343b6f84f4536e/models.py#L29).

* **Caution**: make sure the path is correct. If it is not, it will not be able to load any data. 

### models.py

* **Function**: defines the model architecture and layer features. The model has UNet architecture. 

    [weighted_categorical_crossentropy loss function](https://github.com/aritzLizoain/Image-segmentation/blob/2bd248e3c63bdad6823edbf883343b6f84f4536e/models.py#L29)
    function is used to calculate the categorical crossentropy loss of the model with the addition of taking into account the weight of each class.

#### More information regarding CNNs, UNet, layers, hyperparameter optimization, etc.
* https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
* https://medium.com/intuitive-deep-learning/intuitive-deep-learning-part-2-cnns-for-computer-vision-24992d050a27
* https://medium.com/@jcrispis56/introducci%C3%B3n-al-deep-learning-parte-2-redes-neuronales-convolucionales-f743266d22a0 (Spanish)
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

  * get_monochrome (link) function converts the input image into a monochrome image.
    
Input shape = (n_img, h, w, 3(rgb)) --> Output shape = (n_img, h, w, 1)

after all functions

#Converts the input image (n_img, h, w, 3(rgb)) in (n_img, h, w, 1)
def get_monochrome (images):


PROCESS:    
    *(n_images, x size, y size, 3(rgb)) -------- Takes training image dataset
    
    *(n_images, x size, y size, 1(mean)) ------- Makes it monochrome inside get_mask
    
    *(n_images, x size, y size, n_classes) ----- Creates the mask with get_class in get_mask
    Checks for threshold pixel values 
    E.g.: [background, glowing, hot pixel, cluster] --> n_classes = 4
    get_mask checks every pixel and for each pixel get_class determines the class
    For example for a hot pixel, get_class gets class=2, then n_classes=[0,0,1,0]
    
    *(n_images, x size, y size, 1(max_mask)) --- Gets the maximum value position in mask
    With the previous example: max_mask=2
    
    *(n_images, x size, y size, 3(rgb)) -------- Creates the label with mask_to_label
    Depending on which class it is, it will color it with the corresponding multiplier
    
It plots a random example and shows statistics (n_classes and percentage of each class) 
Different functions are created to convert data, for example: output_to_label()






* **Caution**: it is important to be aware of a possible issue regarding the color of the elements.
The way this model is implemented, image lables do not need to be provided. Image labels are directly obtained from the images.
In order to do this, image pixel values, i.e., colors, are taken as reference to label different classes.
Therefore a color change of an object in the image can cause a wrong label creation if this has not been correctly specified in [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py).







Primero clasifico cada píxel en una de las clases mediante thresholds que corresponden al valor de píxel. Es decir, el color. Entiendo que esto es lo que queríamos hacer con las energías, en caso de que el color esté relacionado con la energía. De hecho sería más fácil, ya que en este he tenido que mirar qué valores de pixel corresponden a cada color, y hay valores que se mezclan. Dado que por ejemplo un pixel con valor 78 es a veces parte de un cluster, y a veces de un hot pixel, los label no son 100% correctos. Pero también es interesante ver si después la predicción es capaz de corregir estos píxeles. La idea de cómo funcionan las labels está muy bien explicado en https://www.jeremyjordan.me/semantic-segmentation/#advanced_unet. Tambien tengo dos imágenes, 'labels' y 'labels2', donde se ve cómo se clasifica cada clase. La mejor parte de esto es que ya no hago los labels como antes, que los hacía a mano con el ratón. Ahora puedo utilizar la cantidad de imágenes que quiera para entrenar.




Idea explained in: https://www.jeremyjordan.me/semantic-segmentation/#advanced_unet

images

it can still be done with labelme or other labeling program









    
### augmentation.py

-**Function**:

-**Caution**:

-**Requirements**:

only geometric. can be added. check [imgaug documentation](https://imgaug.readthedocs.io/en/latest/source/examples_basics.html)

crea más imágenes a partir de las que le paso. Sólo aplico cambios geométricos (rotación, zoom, etc.). Aún así, como ahora puedo crear la cantidad de imágenes que quiera, no lo estoy utilizando. Pero bueno, ahí está.

Here is an augmented image and label example:
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Augmentation_example.png "Augmentation example")

### train.py 

-**Function**:

-**Caution**:

-**Requirements**:

CALLBACKS  https://keras.io/callbacks/  

es el archivo principal, dónde se entrena el modelo. Utiliza las distintas funciones de los archivos que he comentado. En él se puede configurar todo; las rutas de las imágenes, dónde guardar los resultados, características del modelo (por ejemplo que optimizador usar), etc. En la parte final se evalúa el modelo y he puesto un classification report, que es bastante útil para ver cómo ha funcionado cada clase. Os paso un fichero de texto llamado 'console' mostrando un ejemplo de lo que va apareciendo en pantalla al correr este archivo en spyder. Todas las gráficas que se generan se guardan en la carpeta 'Images/Outputs'. La carpeta 'Models' es donde se guardan los modelos entrenados y los datos de precisión y pérdida en cada época del entrenamiento.

for more information about callbacks check link

### load_model.py

-**Function**:

-**Caution**:

-**Requirements**:

carga el modelo ya entrenado y funciona como la última parte de 'train'. Hace las predicciones, la evaluación y el classification report. 

## How to use it

### Creating the data for training

### Preparing the data for training

You need three folders:

* Image folders. Validation split will be automatic. Test images.

* Savings folder

* Models folder

Explain

### Using the python module

setting model settings. more/less layers. hyperparameters

training model

You can import keras_segmentation in  your python script and use the API

```python
from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=51 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)
```

Example of the console display while training:
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/console.gif "Training console display")

-**Caution**: can be rather slow.

### evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )


### More things that I can do with my model...

## Results

The model is able to predict correctly 3 classes: background, glowing and hot pixels. It fails to predict any cluster.

The final accuracy is >99%.

Example results of test images predicted by the model:
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Test_1.png "Test image prediction 1")
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Test_2.png "Test image prediction 2")

The following classification report is obtained:

Class        | Precision| Recall| F1-score| Support
---          | ---      | ---   | ---     | ---
Background   | 1.00     | 1.00  | 1.00    | 60793
Glowing      | 1.00     | 1.00  | 1.00    | 4352
Hot pixel    | 1.00     | 0.95  | 0.97    | 291
Cluster      | 0.00     | 0.00  | 0.00    | 100
**Accuracy**     |          |       | 1.00    | 65536
**Macro avg.**   | 0.75     | 0.74  | 0.74    | 65536
**Weighted avg.**| 1.00     | 1.00  | 1.00    | 65536

## Next steps

### Testing the trained model on real DAMIC detector images.

Real detector images will be tested.

## Contributing

Feel free to submit pull requests.

Please read [CONTRIBUTING.md](https://github.com/aritzLizoain/Image-segmentation/blob/master/CONTRIBUTING.md) for details on the code of conduct, and the process for submitting pull requests.

If you use this code in a publicly available project, please post an issue or create a pull request and your project link will be added here.

## Versioning

Please check on [releases](https://github.com/aritzLizoain/Image-segmentation/releases) to find previous versions of the project.

## Acknowledgments

* Agustín Lantero for the [image_detais.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/image_details.py) and [image_simulation.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/image_simulation.py) codes.
* Rocío Vilar and Alicia Calderón for the help and support. 

## Copyright

Copyright 2020, Aritz Lizoain, All rights reserved.
