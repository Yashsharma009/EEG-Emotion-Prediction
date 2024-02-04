# Emotion Recognition Model

## Overview

This repository contains a TensorFlow-based emotion recognition model using GRU (Gated Recurrent Unit) and features from an emotions dataset. The dataset includes various audio features like mean, FFT, etc., and the model aims to classify emotions into three categories: NEGATIVE, NEUTRAL, and POSITIVE.

## Dataset

The dataset (`emotions.csv`) includes the following features:

- mean_0_a, mean_1_a, ..., fft_749_b
- Labels: NEGATIVE, NEUTRAL, POSITIVE

## Data Preprocessing

The data is preprocessed using a label mapping, and it is split into training and testing sets.

## Model Architecture

The model architecture consists of:

- Input layer with the shape of the training data
- Expanding dimensions using `tf.expand_dims`
- GRU layer with 256 units and return sequences set to True
- Flattening layer
- Dense output layer with softmax activation for three classes (NEGATIVE, NEUTRAL, POSITIVE)

## Training

The model is compiled using the Adam optimizer and sparse categorical crossentropy loss. It is trained for 50 epochs with early stopping. The training history is stored in the `history` variable.

## Evaluation

The model achieves an accuracy of 96.25% on the test set. The confusion matrix and classification report are visualized using seaborn and matplotlib.

## Files

- `emotions.csv`: Raw dataset
- `model_summary.txt`: Summary of the model architecture
- `model_evaluation.txt`: Evaluation metrics of the trained model
- `README.md`: This file

## Usage

1. Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
