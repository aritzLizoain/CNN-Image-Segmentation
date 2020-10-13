# Data Augmentation

Data augmentation is especially useful to teach invariance properties to the network. Augmentation is applied when only a few training samples are available, or when the desired property is not present in the dataset.

The lack of samples does not pose a problem, since the images are simulated. On the other hand, possessing data with certain properties that cannot be simulated can be essential in a project. Labels are augmented together with the images with the purpose of having a larger dataset. These augmentation transformations can include rotations, translations, scaling, cropping, etc.

<p align="center">
<img src="https://github.com/aritzLizoain/CNN-Image-segmentation/blob/master/Images/Example_Images/Augmentation.png" width="400"/>
</p>

*Image and label augmentation example. The applied transformations are translation and scaling.*


