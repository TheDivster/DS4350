# -*- coding: utf-8 -*-
"""Perceptron.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VqRduPWTGvPdDoB1LK-FX1_J9aPjttAB
"""

"""## Load the dataset
Note: we will need to turn the label values into -1 and 1.
"""

import pandas as pd
train = pd.read_csv("/Users/divytripathy/PycharmProjects/Machine Learning/Perceptron/bank-note/train.csv", names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
print(train.head())

train['label'] = train['label'].apply(lambda x: 1 if x == 1 else -1)
print(train.head())

test = pd.read_csv("/Users/divytripathy/PycharmProjects/Machine Learning/Perceptron/bank-note/test.csv", names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
print(test.head())

test['label'] = test['label'].apply(lambda x: 1 if x == 1 else -1)
print(test.head())

"""Our Class:"""

import numpy as np
from sklearn.utils import shuffle

class Perceptron:
  def standard_perceptron(self, X: np.ndarray, Y: np.ndarray, rate: int=1, epoch: int=1):
    """
    This only works when the output is binary in the form of -1 or 1.
    """
    weights: np.ndarray = np.zeros(len(X[0]))
    copy_X = X[:]
    copy_Y = Y[:]
    for _ in range(epoch):
      copy_X, copy_Y = shuffle(copy_X, copy_Y)
      for row, y in zip(copy_X, copy_Y):
        y_pred = np.sign(np.dot(weights, row)).astype(np.int64)
        if y != y_pred:
          weights = weights + rate * (y * row)
    return weights

  def voted_perceptron(self, X: np.ndarray, Y: np.ndarray, rate: int=1, epoch: int=1):
    """
    This only works when the output is binary in the form of -1 or 1.
    returns weights in the form of a list of tuples (weights_i, c_i)
    """
    weights_array: list[tuple] = []
    weights: np.ndarray = np.zeros(len(X[0]))
    c: int = 0

    copy_X = X[:]
    copy_Y = Y[:]

    for _ in range(epoch):
      copy_X, copy_Y = shuffle(copy_X, copy_Y)
      for row, y in zip(copy_X, copy_Y):
        y_pred = np.sign(np.dot(weights, row)).astype(np.int64)
        if y != y_pred:
          weights_array.append((weights, c))
          weights = weights + rate * (y * row)
          c = 1
        else:
          c += 1
    return weights_array

  def average_perceptron(self, X: np.ndarray, Y: np.ndarray, rate: int=1, epoch: int=1):
    """
    This only works when the output is binary in the form of -1 or 1.
    """
    weights: np.ndarray = np.zeros(len(X[0]))
    a: np.ndarray = np.zeros(len(X[0]))
    copy_X = X[:]
    copy_Y = Y[:]
    for _ in range(epoch):
      copy_X, copy_Y = shuffle(copy_X, copy_Y)
      for row, y in zip(copy_X, copy_Y):
        y_pred = np.sign(np.dot(weights, row)).astype(np.int64)
        if y != y_pred:
          weights = weights + rate * (y * row)
        a += weights
    return a

  @staticmethod
  def voted_perceptron_predict(datapoint: np.ndarray, weights: list):
    predictions: float = 0
    for i in range(len(weights)):
      predictions += np.dot(weights[i][0], datapoint) * weights[i][1]
    return np.sign(predictions).astype(np.int32)

  @staticmethod
  def bias_trick(X: np.ndarray):
    """
    Used to add 1 to the input array
    """
    output: list[np.ndarray] = []
    for i in range(len(X)):
      output.append(np.append(X[i], 1))
    return np.array(output)

"""Quick way to check for if bias_trick works."""

array_ex: np.ndarray = np.array([np.array([1, 3, 3]), np.array([2, 3, 3]), np.array([3, 3, 4]), np.array([4, 3, 1])])
print(Perceptron.bias_trick(array_ex))

"""We will look at the accuracy of the standard perceptron."""

print(Perceptron().standard_perceptron(X = Perceptron.bias_trick(train.drop('label', axis=1).values), Y = train['label'].values, epoch=10))

"""Lets make test predictions."""

def accuracy_standard_perceptron(y_test: np.ndarray, x_test: np.ndarray, weights: np.ndarray):
  num_observations: int = len(y_test)
  correct: int = 0
  for i in range(num_observations):
    prediction = np.sign(np.dot(weights, x_test[i]))
    if prediction == y_test[i]:
      correct += 1
  return correct / num_observations

weights = Perceptron().standard_perceptron(X = Perceptron.bias_trick(train.drop('label', axis=1).values), Y = train['label'].values, epoch=10)
print("standard weights: ", weights)
print("train accuracy: ", accuracy_standard_perceptron(train['label'].values, Perceptron.bias_trick(train.drop('label', axis=1).values), weights))
print("test accuracy: ", accuracy_standard_perceptron(test['label'].values, Perceptron.bias_trick(test.drop('label', axis=1).values), weights))

"""Quick check of if everything is as expected."""

weights = Perceptron().voted_perceptron(X = Perceptron.bias_trick(train.drop('label', axis=1).values), Y = train['label'].values, epoch=10)
print(len(weights))

"""Lets test predictions."""

prediction = Perceptron.voted_perceptron_predict(Perceptron.bias_trick(train.drop('label', axis=1).values)[0], weights)
print(prediction, train['label'].values[0])

"""Lets test accuracy."""

def accuracy_voted_perceptron(y_test: np.ndarray, x_test: np.ndarray, weights: list[tuple]):
  num_observations: int = len(y_test)
  correct: int = 0
  for i in range(num_observations):
    prediction = Perceptron.voted_perceptron_predict(x_test[i], weights)
    if prediction == y_test[i]:
      correct += 1
  return correct / num_observations

"""We need to print the count of the weights along with the weights for the voted perceptron."""

weights = Perceptron().voted_perceptron(X = Perceptron.bias_trick(train.drop('label', axis=1).values), Y = train['label'].values, epoch=10)
print("train accuracy: ", accuracy_voted_perceptron(train['label'].values, Perceptron.bias_trick(train.drop('label', axis=1).values), weights))
print("test accuracy: ", accuracy_voted_perceptron(test['label'].values, Perceptron.bias_trick(test.drop('label', axis=1).values), weights))

for weight, count in weights:
  print(count, weight)

"""Lets check the average perceptron accuracy."""

def accuracy_average_perceptron(y_test: np.ndarray, x_test: np.ndarray, weights: np.ndarray):
  num_observations: int = len(y_test)
  correct: int = 0
  for i in range(num_observations):
    prediction = np.sign(np.dot(weights, x_test[i]))
    if prediction == y_test[i]:
      correct += 1
  return correct / num_observations

weights = Perceptron().average_perceptron(X = Perceptron.bias_trick(train.drop('label', axis=1).values), Y = train['label'].values, epoch=10)
print("average perceptron weights: ", weights)
print("train accuracy: ", accuracy_average_perceptron(train['label'].values, Perceptron.bias_trick(train.drop('label', axis=1).values), weights))
print("test accuracy: ", accuracy_average_perceptron(test['label'].values, Perceptron.bias_trick(test.drop('label', axis=1).values), weights))