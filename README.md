
# Image-Classification
## Overview
This project employs neural networks for image classification using a dataset featuring cats and dogs. Designing a custom convolutional neural network for optimal classification and evaluating pre-trained models for comparison.

## Folder Structure
* [ml/pc/exercises:](https://github.com/CaseySobon/Image-classification/tree/main/ml/pc/exercises) This folder has some useful resources used for the project.
* [Cat_vs_dog_CNN.ipynb:](https://github.com/CaseySobon/Image-classification/blob/main/Cat_vs_dog_CNN.ipynb) This file is the self-trained neural network.
* [Cat_vs_dog_resnet50.ipynb:](https://github.com/CaseySobon/Image-classification/blob/main/Cat_vs_dog_resnet50.ipynb) This file is the test of the pre-trained model Resnet50
* [Cat_vs_dog_selftrained_VGG16.ipynb:](https://github.com/CaseySobon/Image-classification/blob/main/Cat_vs_dog_selftrained_VGG16.ipynb) This file is the test for the pre-trained model VGG16.
* [DL_Project_3_Dog_vs_Cat_Classification_Transfer_Learning.ipynb:](https://github.com/CaseySobon/Image-classification/blob/main/Cat_vs_dog_selftrained_VGG16.ipynb) This file is the test for transfer learning.

## Data Analysis

The data set contains 25000 images with 12500 dogs and 12500 cats.
The data will split randomly to have 20000 images for training and 5000 images for validation.

## Convolutional Neural Network (CNN) Classifier
#### Dependencies
* Python 
* TensorFlow
* Keras
* Matplotlib
* NumPy

#### 1. Data Preparation:
* The code starts by mounting Google Drive to access data files stored there.
* It unzips two sets of images, one for training (train.zip) and one for testing (test1.zip).
* The training images are organized into folders for dogs and cats.
* The code uses the ImageDataGenerator from Keras to perform real-time data augmentation and normalization on the training data. It rescales pixel values to the range [0, 1].
* The data generators are created for both training and validation sets.
#### 2. Neural Network Model:
* The neural network model is defined using the Sequential API from Keras.
* It consists of convolutional layers with max pooling, followed by fully connected layers.
* The neural network model defined in the code has the following layers:
  1. Conv2D Layer 1: 32 filters, kernel size 3x3, 'same' padding, ReLU activation.
  2. Conv2D Layer 2: 32 filters, kernel size 3x3, 'same' padding, ReLU activation.
  3. MaxPooling2D Layer 1: Max pooling with default 2x2 pool size.
  4. Conv2D Layer 3: 64 filters, kernel size 3x3, 'same' padding, ReLU activation.
  5. Conv2D Layer 4: 64 filters, kernel size 3x3, 'same' padding, ReLU activation.
  6. MaxPooling2D Layer 2: Max pooling with default 2x2 pool size.
  7. Conv2D Layer 5: 128 filters, kernel size 3x3, 'same' padding, ReLU activation.
  8. Conv2D Layer 6: 128 filters, kernel size 3x3, 'same' padding, ReLU activation.
  9. MaxPooling2D Layer 3: Max pooling with default 2x2 pool size.
  10. Flatten Layer: Flattens the input.
  11. Dense Layer 1: 256 units, ReLU activation.
  12. Dropout Layer 1: Dropout with a rate of 0.5.
  13. Dense Layer 2: 256 units, ReLU activation.
  14. Dropout Layer 2: Dropout with a rate of 0.5.
  16. Dense Output Layer: 1 unit with a sigmoid activation function.
*The activation function used is ReLU for convolutional layers and sigmoid for the output layer.
* Dropout layers are added to reduce overfitting.
#### 3. Model Compilation:
* The model is compiled using the Adam optimizer and binary cross-entropy loss, as it's a binary classification problem.
* The accuracy metric is used to monitor the model's performance during training.
#### 4. Model Training:
* The model is trained using the fit method with the training and validation data generators.
* Training progress is printed out for each epoch, showing training and validation loss and accuracy.
5. Model Evaluation and Visualization:
* The code visualizes the model architecture using plot_model.
* The training history is plotted, showing training and validation accuracy and loss over epochs.
![plot1](https://github.com/CaseySobon/Image-classification/assets/96227583/c15d3cbe-cb5e-4ec4-a5c5-9fae12812719)
#### 6. Model Testing:
* Another set of images (test set) is prepared using an ImageDataGenerator.
* The trained model is used to make predictions on the test set.

## Pre-trained Models
#### Dependencies
* Python 3.x
* TensorFlow
* Keras
* Matplotlib
* NumPy
#### Implementing Pre-trained ResNet50
1. Import Necessary Libraries
  * from keras.applications import ResNet50
  * from keras.applications.resnet50 import preprocess_input
2. Load Pre-trained ResNet50 Model
3. Freeze Pre-trained Layers
4. Recreate a New Model
5. Compile the Model
6. Train the Model
7. Evaluate and Visualize

<img src=(https://github.com/CaseySobon/Image-classification/assets/96227583/70076607-eb31-433e-9f62-7b841da36bde width="300">
8. Test the Model
#### Implementing Pre-trained VGG16
This is done in the same way as before but with the necessary libraries.
1. Import Necessary Libraries
  * from keras.applications import VGG16
  * from keras.applications.VGG16 import preprocess_input
  * ....
6. ....
7. Then Evaluate and Visualize
![plot3](https://github.com/CaseySobon/Image-classification/assets/96227583/e12807af-a5b7-463d-8d5c-bf06b010fc10)
8. Test the Model









