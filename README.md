
# Image-Classification
## Overview
This project is to explore ways to use neural networks to do image classification. I am using a dataset that has images of cats and dogs and will find the best method to classify them. There will be a convolutional neural network I will design myself to see how well I can do it. Then will try the available pre-trained models.

## Folder Structure
* cnn_classifier/: Contains the code for the CNN-based image classifier.
* pretrained_models/: Includes scripts to utilize pre-trained models for image classification.


## Data Analysis


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
