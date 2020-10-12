# Model Training

Two datasets are created: the training and testing sets. As a matter of fact, the training set is splitted into a training and a validation set. The validation set can be understood as a testing set that is applied while training. It is essential to verify that the network has not memorized the training data (i.e. overfitting) and will later be able to reliably perform on the 'unseen' test dataset.

The training dataset is shuffled and divided into batches. Then, these batches were passed through the network a certain number of times, defined as epochs. A small batch size introduces a high variation within each batch, as it is improbable that a small number of training samples represent the dataset reasonably. Nonetheless, choosing a large batch size may tend to overfit the data. During the training process, the model is expected to stabilize at its optimum state, and converge its loss and accuracy.

<pre>
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Accuracy.png" width="400"/>           <img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/Loss.png" width="400"/> 
</pre>

*Example of a correct training of a model that reaches its optimum state at epoch 40.* <br\> LEFT: *Training and validation accuracy.* <br/> *RIGHT: Training and validation loss.*

However, models can occasionally be overtrained after reaching their optimum state and cause overfitting. In order to avoid this, the training process is programmed to stop if the validation loss has not improved in 20 epochs, and only the model with the lowest loss is saved.

<pre>
<img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/AccuracyOverfit.png" width="400"/>           <img src="https://github.com/aritzLizoain/Image-segmentation/blob/master/Images/Example_Images/LossOverfit.png" width="400"/> 
</pre>

*Example of an overtrained model that overfits after reaching its optimum state at epoch 25.* <br\> LEFT: *Training and validation accuracy.* <br/> *RIGHT: Training and validation loss.*


