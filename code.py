# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv(r'C:\Users\91934\Downloads\archive (3)\data.csv')

# Create target variable: 1 for malignant (M), 0 for benign (B)
df['target'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Drop unnecessary columns
df.drop(['diagnosis', 'id'], axis=1, inplace=True)

# Display dataset information
df.info()

# Display statistical summary of the dataset
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Split the dataset into features (X) and target (Y)
X = df.drop('target', axis=1)
Y = df['target']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the logistic regression model
reg = LogisticRegression()

# Define hyperparameters for RandomizedSearchCV
param_distributions = {'C': [1, 2, 3, 4, 5, 10, 100], 'max_iter': [50, 100, 150]}

# Perform hyperparameter tuning using RandomizedSearchCV
model = RandomizedSearchCV(reg, param_distributions, cv=5, scoring='f1')
model.fit(x_train, y_train)

# Display the best score obtained from hyperparameter tuning
print(f'Best F1 Score from RandomizedSearchCV: {model.best_score_:.2f}')

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate and display the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy = {accuracy:.2f}')

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Display the classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
