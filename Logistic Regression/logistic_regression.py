#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import os

cwd = os.getcwd()
train = pd.read_csv(cwd + "/bank-note/train.csv", names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
print(train.head())


# In[41]:


train['label'] = train['label'].apply(lambda x: 1 if x == 1 else -1)
print(train.head())


# In[42]:


test = pd.read_csv(cwd + "/bank-note/test.csv", names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
print(test.head())


# In[43]:


test['label'] = test['label'].apply(lambda x: 1 if x == 1 else -1)
print(test.head())


# In[44]:


def bias_trick(X: np.ndarray):
  """
  Used to add 1 to the input array
  """
  output: list[np.ndarray] = []
  for i in range(len(X)):
    output.append(np.append(X[i], 1))
  return np.array(output)

X = train.values[:,:-1]
y = train.values[:, -1].astype(int)
X_val = test.values[:,:-1]
y_val = test.values[:, -1].astype(int)
X_bias = bias_trick(X)
X_val_bias = bias_trick(X_val)
print(X_bias)


# Note: Remove the w to report just MLE

# In[48]:


import sklearn
from numpy import linalg as LN
import math

epochs = 100
y = y.astype(float)
weights_array = []
for var in [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]:
    weights = np.zeros(X_bias.shape[1]).astype(float)
    learning_rate = 3e-4
    for epoch in range(epochs):
        sklearn.utils.shuffle(X_bias, y)
        for i in range(X_bias.shape[0]):
            learning_rate = learning_rate / (1 + learning_rate/1000000 * i)
            print(LN.norm((-1 * y[i] * X_bias[i] * math.exp(-1 * y[i] * np.dot(weights, X_bias[i])) / (1 + math.exp(-1 * y[i] * np.dot(weights, X_bias[i]))) + var * weights)))
            step = learning_rate * (-1 * y[i] * X_bias[i] * math.exp(-1 * y[i] * np.dot(weights, X_bias[i])) / (1 + math.exp(-1 * y[i] * np.dot(weights, X_bias[i]))) + var * weights)
            weights = weights - step
    weights_array.append((var, weights))
print(weights)


# In[49]:


for i, weights in weights_array:
    count = 0
    for example, answer in zip(X_bias, y):
        if np.sign(np.dot(weights, example)) == answer:
            count += 1
    print("var", i, "train accuracy", count / len(y))    


# In[50]:


for i, weights in weights_array:
    count = 0
    for example, answer in zip(X_val_bias, y_val):
        if np.sign(np.dot(weights, example)) == answer:
            count += 1
    print("var", i, "test accuracy", count / len(y)) 


# In[ ]:


get_ipython().system('jupyter nbconvert --to script logistic_regression.ipynb')

