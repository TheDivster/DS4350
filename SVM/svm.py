#!/usr/bin/env python
# coding: utf-8

# In[12]:


# from google.colab import drive
# drive.mount("/content/gdrive")


# 

# ## Load the dataset
# Note: we will need to turn the label values into -1 and 1.

# In[13]:


import pandas as pd
import os

cwd = os.getcwd()
train = pd.read_csv(cwd + "/bank-note/train.csv", names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
print(train.head())


# In[ ]:


train['label'] = train['label'].apply(lambda x: 1 if x == 1 else -1)
print(train.head())


# In[ ]:


test = pd.read_csv(cwd + "/bank-note/test.csv", names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
print(test.head())


# In[ ]:


test['label'] = test['label'].apply(lambda x: 1 if x == 1 else -1)
print(test.head())


# Our Class:

# In[ ]:


import numpy as np
from sklearn.utils import shuffle

class svm:
  @staticmethod
  def primal_svm(X: np.ndarray, Y: np.ndarray, C = 1, rate_0: int=1, epoch: int=1):
    """
    This only works when the output is binary in the form of -1 or 1.
    """
    weights: np.ndarray = np.zeros(len(X[0]))
    w_0 = np.ndarray = np.zeros(len(X[0]))
    copy_X = X[:]
    copy_Y = Y[:]
    N: int = X.shape[0]
    alpha = 0.5
    counter = 0
    for _ in range(epoch):
      copy_X, copy_Y = shuffle(copy_X, copy_Y)
      for row, y in zip(copy_X, copy_Y):
        y_pred = y * np.sign(np.dot(weights, row)).astype(np.int64)
        rate = rate_0 / (1 + (rate_0/alpha) * counter)
        if y_pred <= 1:
          weights = weights - rate * w_0 - rate * N * C * (y * row)
        else:
          w_0 = (1 - rate) * w_0
          # weights = weights - rate * w_0
        counter += 1
    return weights

  @staticmethod
  def bias_trick(X: np.ndarray):
    """
    Used to add 1 to the input array
    """
    output: list[np.ndarray] = []
    for i in range(len(X)):
      output.append(np.append(X[i], 1))
    return np.array(output)


# We will now look at the accuracy of the primal SVM.

# In[ ]:


# print(svm().primal_svm(X = svm.bias_trick(train.drop('label', axis=1).values), Y = train['label'].values, epoch=10))


# Lets make predictions:

# In[ ]:


def accuracy_primal_svm(y_test: np.ndarray, x_test: np.ndarray, weights: np.ndarray):
  num_observations: int = len(y_test)
  correct: int = 0
  for i in range(num_observations):
    prediction = np.sign(np.dot(weights, x_test[i]))
    if prediction == y_test[i]:
      correct += 1
  return correct / num_observations


# In[ ]:


weights = np.array([10371.27739925, 14676.61815586,  -603.08069587, -1642.48309612,
  1246.37583512])
print("train accuracy: ", accuracy_primal_svm(train['label'].values, svm.bias_trick(train.drop('label', axis=1).values), weights))
print("test accuracy: ", accuracy_primal_svm(test['label'].values, svm.bias_trick(test.drop('label', axis=1).values), weights))


# In[ ]:


# from sklearn.datasets import make_classification
# from scipy.optimize import minimize

# y = train['label'].values
# X = train.drop('label', axis=1, inplace=False).values
# def dual_objective(alpha, X, y):
#     return 0.5 * np.dot(alpha, alpha) - np.sum(alpha * y * np.dot(X, X.T).dot(y))  # SVM dual objective function

# constraints = ({'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y), 'jac': lambda alpha: y})

# bounds = [(0, None) for _ in range(len(y))]

# alpha_init = np.zeros(len(y))

# result = minimize(dual_objective, alpha_init, args=(X, y), method='SLSQP', bounds=bounds, constraints=constraints)

# alpha_optimized = result.x

# support_vector_indices = np.where(alpha_optimized > 1e-5)[0]
# support_vectors = X[support_vector_indices]
# support_vector_labels = y[support_vector_indices]

# weights = np.sum(alpha_optimized * support_vector_labels * support_vectors.T, axis=1)
# bias = np.mean(support_vector_labels - np.dot(support_vectors, weights))

# print("Optimized Lagrange multipliers (alpha):", alpha_optimized)
# print("Weights:", weights)
# print("Bias:", bias)


# In[ ]:


weights_1 = svm.primal_svm(svm.bias_trick(train.drop('label', axis=1, inplace=False).values), train['label'].values, C = 700/873, epoch=100)
print("standard weights: ", weights_1)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script svm.ipynb')

