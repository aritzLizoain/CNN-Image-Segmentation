# CNN UNet Image Segmentation

The **goal** of the project is to study machine learning techniques on detector images from Beyond Standard Model searches.
The studied images represent energy deposits on CCDs.

>Image segmentation clusters pixels into salient image regions, i.e., regions corresponding to individual surfaces,objects, or natural parts of objects.

This is a **multiclass image segmentation** project. UNet structure has been used for the model.

image of goal

## Getting Started

These instructions explain how to get a copy of the project, run it on your local machine for development and testing purposes.

### Prerequisites

You will need to install the following libraries:

Used libraries: keras, scikit-learn, opencv, etc. (version numbers are up to date: 19.06.2020)

### Installing

Installing, downloading, cloning the project:

How to install libraries/modules/how are they called:

```
open anaconda prompt
```

```
activate environment
```

```
pip install 'name'
```

For all libraries that are not already installed

## Explanation of each code. How they work

### image_details.py

-**Function**: set details of the simulated images that are created in image_simulation.py.

-**Author**: Agustín Lantero.

### image_simulation.py

-**Function**: create simulated images. Parameters such as number of elements, noise, glowing, images, etc. can be defined.
Images are saved to the saving path.

-**Caution**: it is important to be aware of a possible issue regarding the color of the elements.
The way this model is implemented, image lables do not need to be provided. Image labels are directly obtained from the images.
In order to do this, image pixel values, i.e., colors, are taken as reference to label different classes (read [mask.py information](https://github.com/aritzLizoain/Image-segmentation#maskpy) for more information).
Therefore a color change of an object in the image can cause a wrong label creation if this has not been correctly specified in [mask.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/mask.py).

-**Requirements**: working directory path must contain image_details.py and Cluster.pkl.

-**Author**: Agustín Lantero.

### load_dataset.py

-**Function**: takes images and saves them as numpy arrays. 

[get_weights](https://github.com/aritzLizoain/Image-segmentation/blob/0fc6f36abc9fcc63aee3c5129989fff54891147e/load_dataset.py#L52)
function is used to avoid issues with imbalanced datasets, where images are too biased towards one class.
In this case ~95% of the pixels are labeled as background, and only ~0.1% as clusters. 
This way the model can give a 95% accuracy prediction, but the predicted label will be all black, predicted as background.
The weight of each class is obtained as the inverse of its presence percentage over all the training samples.
Then the weights are normalized to the number of classes.
These weights are used by the model in the [weighted_categorical_crossentropy loss function](https://github.com/aritzLizoain/Image-segmentation/blob/2bd248e3c63bdad6823edbf883343b6f84f4536e/models.py#L29).

-**Caution**: make sure the path is correct. If it is not, it will not be able to load any data. 

### models.py

* **Function**:

* **Caution**:

* **Requirements**:

things that can be changed (or better on how to use)
maybe I just need to explain what it has and the options that they give. And then on how to use I can explain how to use it differently

contiene la arquitectura del modelo. Un UNet normal, parecido al que tenía Alicia en su código. Tengo muchas capas escritas, pero algunas están comentadas. Como os lo mando es lo que me ha dado mejor resultado hasta ahora. Y tambien defino la función de pérdida que utiliza los pesos de cada clase. Esta función de pérdida la encontré en internet. Es un 'categorical_cossentropy' con el extra de que tiene en cuenta los pesos.

What is **CNN**
>Definition

![alt text](https://miro.medium.com/max/1000/1*zNs_mYOAgHpt3WxbYa7fnw.png "CNN example")

What is **UNet**
>Definition

Reference images
Image: 
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Models/Architecture%201.png "UNet architecture 1")
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Models/Architecture%202.png "UNet architecture 2")

CNNs https://medium.com/intuitive-deep-learning/intuitive-deep-learning-part-2-cnns-for-computer-vision-24992d050a27
    
Convolution layer: uses parameter sharing and apply the same smaller set of parameters spatially across the image (reduces the number of parameters needed). Hyper-parameter filter size corresponds to how many input features in the width and height dimensions one neuron takes in. Stride helps overlapping regions, it defines how many pixels we want to move when apply the neuron again. Depth defines how many different outputs do we have (cat + dog = 2). If output dimensions 254*254*64 and we want 256*256*64, padding will add a border of 0s.
    
Max Pool layer: takes the maximum of the numbers it looks at. Applies to each individual depth channel separately; leaves depth dimension unchanged. Defined only by filter size and stride, reducing the spatial size by taking the maximum of the numbers within its filter.
   
Dropout layer (not in the article but in the example): like in house price prediction, dropout from keras.layers for avoiding overfitting.
    
Fully-Connected (FC) layer: same as our standard neural network; every neuron in the next layer takes as input every neuron in the previous layer's output. Usually used at the end of the CNNs. We can flatten the neurons into a one-dimensional array of features. Apply hidden layers as per usual.
    
Softmax layer (not in the article but in the example): last layer, softmax, only transforms the output of the previous layer into probability distributions, which is the final goal.

INTRODUCCIÓN CNNs https://medium.com/@jcrispis56/introducci%C3%B3n-al-deep-learning-parte-2-redes-neuronales-convolucionales-f743266d22a0
    
Similar to item 3. Convolución, función rectificadora (ReLU), Max Pooling, conexión total, softmax y cross entropy (binary). Evita overfitting aplicando transformaciones, como rotaciones, zoom e inversiones en el eje horizontal, y rescale.
    
BEGINNER GUIDE CNNs https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
    
Classic CNN architecture: input-conv-ReLU-conv-ReLU-max pool-ReLU-conv-ReLU-pool-(flatten)fc.
Recommended paper: Visualizing and Understanding
Convolutional Networks.
Training process backpropagation, loss function.

CNN GUIDE https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

Convolution layer (filter same depth as channels RGB) (output = channel 1 + channel 2 + channel 3 + bias(=1)) + padding, Max Pooling layer (decrease the computational power required to process the data through dimensionality reduction)(performs better than Average Pooling), flatten, fully connected (ReLU activation), softmax.

CONVOLUTIONAL NEURAL NETWORKS FOR BEGINNERS 
%https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca
https://towardsdatascience.com/ convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca

Hyperparameters etc. explained

SEMANTIC SEGMENTATION U-NET https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47 https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb
     
CALLBACKS  https://keras.io/callbacks/  
     
Up sampling with UpConv=Conv2DTranspose to perform up sampling; from low to high resolution image. Learns parameters through back propagation to convert. Needs concatenate with corresponding layer.
 
### mask.py

-**Function**:

-**Caution**:

-**Requirements**:

es lo más interesante. Aquí creo los labels de todas las imágenes. Primero clasifico cada píxel en una de las clases mediante thresholds que corresponden al valor de píxel. Es decir, el color. Entiendo que esto es lo que queríamos hacer con las energías, en caso de que el color esté relacionado con la energía. De hecho sería más fácil, ya que en este he tenido que mirar qué valores de pixel corresponden a cada color, y hay valores que se mezclan. Dado que por ejemplo un pixel con valor 78 es a veces parte de un cluster, y a veces de un hot pixel, los label no son 100% correctos. Pero también es interesante ver si después la predicción es capaz de corregir estos píxeles. La idea de cómo funcionan las labels está muy bien explicado en https://www.jeremyjordan.me/semantic-segmentation/#advanced_unet. Tambien tengo dos imágenes, 'labels' y 'labels2', donde se ve cómo se clasifica cada clase. La mejor parte de esto es que ya no hago los labels como antes, que los hacía a mano con el ratón. Ahora puedo utilizar la cantidad de imágenes que quiera para entrenar.

it can still be done with labelme or other labeling program
    
### augmentation.py

-**Function**:

-**Caution**:

-**Requirements**:

only geometric. can be added. check [imgaug documentation](https://imgaug.readthedocs.io/en/latest/source/examples_basics.html)

crea más imágenes a partir de las que le paso. Sólo aplico cambios geométricos (rotación, zoom, etc.). Aún así, como ahora puedo crear la cantidad de imágenes que quiera, no lo estoy utilizando. Pero bueno, ahí está.

Here is an augmented image and label example:
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/Augmentation_example.png "Augmentation example")

### train.py 

-**Function**:

-**Caution**:

-**Requirements**:

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
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/console.gif "Training console display")

-**Caution**: can be rather slow.

### evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )


### More things that I can do with my model...

## Results

The model is able to predict correctly 3 classes: background, glowing and hot pixels. It fails to predict any cluster.

The final accuracy is >99%.

Example results of test images predicted by the model:
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/Test_1.png "Test image prediction 1")
![alt text](https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Outputs/Test_2.png "Test image prediction 2")

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
