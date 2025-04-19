# Digit Classifier using CNN & FastAPI

##  Project Overview
This project implements a digit classification system using a Convolutional Neural Network (CNN) trained on the kaggle digit recognizer competition dataset which is a subset of the MNIST dataset. It classifies handwritten digits (0–9) with high accuracy and is deployed as a web API using FastAPI for real-time predictions.

The pipeline covers:

- Data loading, visualization, and preprocessing
- CNN architecture building, training, and evaluation
- Performance visualization and prediction output
- Feature map visualizations from convolutional layers
- Deployment using FastAPI for serving predictions via HTTP requests

## Dataset
Kaggle digit recognizer competition dataset which is a subset of the MNIST dataset (https://www.kaggle.com/competitions/digit-recognizer/data)
The CSV file contains 42000 rows/samples with 784 columns(28x28)handwritten digit images as pixel intensity.
This is split for training and testing (80% and 20%).

## Setup Instructions
- Install all the required libraries and data file
- Train and save the model
- Run the fast api app
- Upload a handwritten image to test
 
