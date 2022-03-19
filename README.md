quickdraw-torch-cnn
-------------------

This is a Convolutional Neural Network classification implementation for Google's _"Quick, Draw!"_ [dataset](https://github.com/googlecreativelab/quickdraw-dataset). The instructions outlined here are primarily for OSX systems (even runs on an M1), but could be ported reasonably easy to other unix* operating systems.

# Overview

This repository contains the code for training, evaluating, and using a CNN to classify drawings from the QuickDraw dataset. The outline for this repository is given below:

1. [CNN architecture](#cnn-architecture)
2. [Obtaining the data](#obtaining-the-data)
3. [Training a classifier](#training-a-classifier)
4. [Evaluation](#evaluation)
5. [Inference](#inference)

# CNN architecture

This implementation follows the suggestion by [Chandra Kanth](https://github.com/ck090) and reflects a standard approach to image classification tasks. The CNN is as follows:

- A convolutional layer of size 5x5
- Max pooling of size 2x2
- Hidden convolutional layer of size 2x2
- Hidden max pooling of size 2x2
- Dropout layer with probability of 20%
- Flatten layer
- Fully-connected layer with 128 units and ReLu activation
- Fully-connected layer with 50 units and ReLu activation
- Output layer (classification)

# Obtaining the data

Downloads the data using `gsutil`. You can specify the name of individual files instead of a wildcard, if you are not interested in training all `345` categories.

1. `brew install --cask google-cloud-sdk`
2. `cd data`
3. `gsutil -m cp 'gs://quickdraw_dataset/full/numpy_bitmap/*.npy' .`

# Training a classifier

Creates a virtual environment and installs the necessary dependencies, proceeds to train the CNN. Make sure you have sufficient amount of memory at your disposal, otherwise this script will likely be killed by your OS.

1. `python3 -m venv venv && source venv/bin/activate` (if you use `pyenv` or `poetry`, you know what to do here)
2. `pip install -r requirements.txt`
3. `python3 cnn/train_cnn.py` (run `--help` if you want to see available parameters)

# Evaluation



# Inference