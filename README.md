# CNN U-Net Image Segmentation

check from overleaf as well

give links to check more about layers, optimizers, etc.

also take the information that I already have

The **goal** of the project is to study machine learning techniques on detector images from Beyond Standard Model searches.

The images represent energy deposits on CCDs.

This is a **multiclass image segmentation** project.

>Image segmentation clusters pixels into salient image regions, i.e., regions corresponding to individual surfaces,objects, or natural parts of objects.


One of the main applications of image segmentation is developing self-driving cars: 
![alt text](https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/opencv-semantic-segmentation/opencv_semantic_segmentation_animation.gif "Segmentation example")

What is **CNN**
>Definition

![alt text](https://miro.medium.com/max/1000/1*zNs_mYOAgHpt3WxbYa7fnw.png "CNN example")


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

los códigos de Agustín para generar las imágenes. Las imágenes creadas las guardo en la carpeta de 'Images/Train' e 'Images/Test'. Nada nuevo.

take into account that the way the code is implemented, the automatic labels will be created depending on colors of these created images.
Meaning that if same class has different colour, the label will need to be created in another way, for example, with labelme
For the purpose of this segmentation, colour works well.

### image_simulation.py

Make sure that directory path contains image_details.py

Also saving path can be changed but will be used later

los códigos de Agustín para generar las imágenes. Las imágenes creadas las guardo en la carpeta de 'Images/Train' e 'Images/Test'. Nada nuevo.

### load_dataset.py

make sure path is correct
anything else to take into account?
simplemente lee las imágenes y las guarda en arrays. Y luego hay una función llamada 'get_weights'. Lo que me pasa con estas imágenes es que aproximadamente 95% de los píxeles son de la categoría background, y por ejemplo los clusters son sólo un 0.1%. Esto hacía que cuando entrenaba el modelo me daba una precisión del 95%, pero lo único que hacía era decirme que todo es background. La forma de arreglar esto fue dándoles pesos a las clases, que se calculan como la inversa de la frecuencia con la que aparecen en las imágenes. Estos pesos los utilizo en la función de pérdida.

### models.py

things that can be changed (or better on how to use)
maybe I just need to explain what it has and the options that they give. And then on how to use I can explain how to use it differently

contiene la arquitectura del modelo. Un UNet normal, parecido al que tenía Alicia en su código. Tengo muchas capas escritas, pero algunas están comentadas. Como os lo mando es lo que me ha dado mejor resultado hasta ahora. Y tambien defino la función de pérdida que utiliza los pesos de cada clase. Esta función de pérdida la encontré en internet. Es un 'categorical_cossentropy' con el extra de que tiene en cuenta los pesos.

What is **UNet**
>Definition

Image: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

    
### mask.py

es lo más interesante. Aquí creo los labels de todas las imágenes. Primero clasifico cada píxel en una de las clases mediante thresholds que corresponden al valor de píxel. Es decir, el color. Entiendo que esto es lo que queríamos hacer con las energías, en caso de que el color esté relacionado con la energía. De hecho sería más fácil, ya que en este he tenido que mirar qué valores de pixel corresponden a cada color, y hay valores que se mezclan. Dado que por ejemplo un pixel con valor 78 es a veces parte de un cluster, y a veces de un hot pixel, los label no son 100% correctos. Pero también es interesante ver si después la predicción es capaz de corregir estos píxeles. La idea de cómo funcionan las labels está muy bien explicado en https://www.jeremyjordan.me/semantic-segmentation/#advanced_unet. Tambien tengo dos imágenes, 'labels' y 'labels2', donde se ve cómo se clasifica cada clase. La mejor parte de esto es que ya no hago los labels como antes, que los hacía a mano con el ratón. Ahora puedo utilizar la cantidad de imágenes que quiera para entrenar.
    
### augmentation.py

only geometric. can be added. check [imgaug documentation](https://imgaug.readthedocs.io/en/latest/source/examples_basics.html)

crea más imágenes a partir de las que le paso. Sólo aplico cambios geométricos (rotación, zoom, etc.). Aún así, como ahora puedo crear la cantidad de imágenes que quiera, no lo estoy utilizando. Pero bueno, ahí está.

### train.py 

es el archivo principal, dónde se entrena el modelo. Utiliza las distintas funciones de los archivos que he comentado. En él se puede configurar todo; las rutas de las imágenes, dónde guardar los resultados, características del modelo (por ejemplo que optimizador usar), etc. En la parte final se evalúa el modelo y he puesto un classification report, que es bastante útil para ver cómo ha funcionado cada clase. Os paso un fichero de texto llamado 'console' mostrando un ejemplo de lo que va apareciendo en pantalla al correr este archivo en spyder. Todas las gráficas que se generan se guardan en la carpeta 'Images/Outputs'. La carpeta 'Models' es donde se guardan los modelos entrenados y los datos de precisión y pérdida en cada época del entrenamiento.

### load_model.py

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

# evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )

```

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

* Agustín Lantrero for the [image_detais.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/image_details.py) and [image_simulation.py](https://github.com/aritzLizoain/Image-segmentation/blob/master/image_simulation.py) codes.
* Rocío Vilar and Alicia Calderón for the help and support. 

## Copyright

Copyright 2020, Aritz Lizoain, All rights reserved.
