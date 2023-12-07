
# Image-Classification
## Overview
This project employs neural networks for image classification using a dataset featuring cats and dogs. Designing a custom convolutional neural network for optimal classification and evaluating pre-trained models for comparison.

## Folder Structure
* cnn_classifier/: Contains the code for the CNN-based image classifier.
* pretrained_models/: Includes scripts to utilize pre-trained models for image classification.


## Data Analysis

The data set contains % images with % dogs and % cats.
The data will is split randomly to have % images for training and % images for testing.

## Convolutional Neural Network (CNN) Classifier
#### Dependencies
* Python 
* TensorFlow
* Keras
* Matplotlib
* NumPy

#### Usage
* Navigate to the cnn_classifier/ directory.
* Install dependencies using pip install -r requirements.txt.
* Run python train_cnn.py to train the CNN on the provided dataset.
* After training, use python predict_cnn.py --image <path_to_image> to classify a new image.

## Pre-trained Models
#### Dependencies
* Python 3.x
* TensorFlow
* Keras
* Matplotlib
* NumPy

#### Usage
* Navigate to the pretrained_models/ directory.
* Install dependencies using pip install -r requirements.txt.
* Run python predict_pretrained.py --image <path_to_image> to use a pre-trained model for image classification.

#### Available Pre-trained Models
* VGG16
* ResNet50
* MobileNetV2
