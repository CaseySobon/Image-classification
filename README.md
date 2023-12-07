
# Image-Classification
## Overview
This project employs neural networks for image classification using a dataset featuring cats and dogs. Designing a custom convolutional neural network for optimal classification and evaluating pre-trained models for comparison.

## Folder Structure
* [ml/pc/exercises:] (https://github.com/CaseySobon/Image-classification/tree/main/ml/pc/exercises) This folder has some useful resources used for the project.
* Cat_vs_dog_CNN.ipynb: This file is the self-trained neural network.
* Cat_vs_dog_resnet50.ipynb: This file is the test of the pre-trained model Resnet50
* Cat_vs_dog_selftrained_VGG16.ipynb: This file is the test for the pre-trained model VGG16.
* DL_Project_3_Dog_vs_Cat_Classification_Transfer_Learning.ipynb: This file is the test for transfer learning.

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
