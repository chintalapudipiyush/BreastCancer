# Breast Cancer Classification using Logistic Regression

This project implements a logistic regression model to classify breast cancer tumors as malignant or benign based on various features derived from digitized images of fine needle aspirate (FNA) of breast mass.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)

## Introduction

Breast cancer is a common cancer among women worldwide. Early detection and diagnosis are crucial for effective treatment. This project uses machine learning techniques to classify breast cancer tumors based on various features.

## Dataset

The dataset used for this project is the [Breast Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/breast-cancer-dataset/data) available on Kaggle. It contains information about various tumors, including their features and corresponding diagnoses.

- **Features:** 30 numeric features (e.g., radius, texture, perimeter, area, smoothness)
- **Target:** Binary classification (Malignant = 1, Benign = 0)

## Data Preprocessing

- The target variable is created with `1` for malignant (M) and `0` for benign (B).
- Unnecessary columns such as `diagnosis` and `id` are dropped from the dataset.
- The dataset is split into features (X) and target (Y).

## Model Training

- The data is split into training and testing sets.
- A logistic regression model is defined and hyperparameter tuning is performed using `RandomizedSearchCV`.
- The model is trained using the preprocessed data.

## Evaluation

- The trained model is evaluated using the test dataset.
- The evaluation metrics used include accuracy, confusion matrix, and classification report.

## Usage

To run the code:

1. Make sure you have Python installed.
2. Install the required libraries using:

## Results

Best F1 Score from RandomizedSearchCV: 0.95

Accuracy = 0.96

Confusion Matrix:
[[70  1]
 [ 3 40]]
 
Classification Report:
|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.96      | 0.99   | 0.97     | 71      |
| 1             | 0.98      | 0.93   | 0.95     | 43      |
| **accuracy**  |           |        | 0.96     | 114     |
| **macro avg** | 0.97      | 0.96   | 0.96     | 114     |
| **weighted avg** | 0.97   | 0.96   | 0.96     | 114     |
