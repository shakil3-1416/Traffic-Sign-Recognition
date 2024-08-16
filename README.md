# Traffic-Sign-Recognition
The Traffic Sign Recognition project uses a Convolutional Neural Network (CNN) to classify traffic signs from the GTSRB dataset. The model is trained with image preprocessing and data augmentation techniques to improve accuracy and robustness. It can be applied in autonomous driving and intelligent transport systems.

## Dataset

The dataset contains 43 different classes of traffic signs. The data is split into training and test sets, with images of varying sizes and qualities.

- **Training Data**: The training dataset consists of images in `.ppm` format stored in the `Final_Training/Images` folder.
- **Test Data**: The test dataset consists of images in `.ppm` format stored in the `Final_Test/Images` folder. Labels for the test data are provided in the `GT-final_test.csv` file.

## Preprocessing

The preprocessing steps applied to the images include:

1. **Histogram Equalization**: Normalize the intensity of the images to handle overexposed or underexposed images.
2. **Resizing**: All images are resized to a fixed dimension of `48x48` pixels to ensure consistency across the dataset.
3. **Color Conversion**: The images are converted to HSV format for enhanced feature extraction.

## Model Architecture

The CNN model is built using the Keras library with the following architecture:

- **6 Convolutional Layers**: With increasing filters of 32, 64, and 128.
- **MaxPooling Layers**: Applied after each set of convolutional layers to reduce the spatial dimensions.
- **Dropout Layers**: Added after each max-pooling layer to prevent overfitting.
- **Fully Connected (Dense) Layers**: After flattening, a dense layer of 512 units is used followed by a softmax layer with 43 units for the final classification.

### Model Summary

1. **Input Layer**: `48x48x3` (RGB images)
2. **Convolutional Layers**: ReLU activation, filters of size `(3x3)`
3. **Pooling Layers**: MaxPooling `(2x2)`
4. **Dropout Layers**: Dropout rates of 0.2 and 0.5 for regularization
5. **Output Layer**: Softmax activation with 43 classes

## Training

The model is compiled using:
- **Loss Function**: `categorical_crossentropy`
- **Optimizer**: Stochastic Gradient Descent (SGD) with Nesterov momentum.
- **Metrics**: Accuracy

The training is performed over 30 epochs with a batch size of 32. Early stopping is used to prevent overfitting, and the learning rate is scheduled to decay by a factor of 0.1 every 10 epochs.

### Data Augmentation

To improve model performance and prevent overfitting, data augmentation is applied to the training images. Augmentations include:
- **Rotation**: Up to 10 degrees.
- **Width and Height Shifts**: Up to 10%.
- **Zoom and Shear Transformations**

## Evaluation

The trained model is evaluated on the test set using accuracy as the performance metric. The final accuracy achieved on the test set is reported after training with data augmentation.


