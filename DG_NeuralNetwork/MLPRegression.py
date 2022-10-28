#!/usr/bin/env python
# coding: utf-8

# ### Multi Layer Perceptron Regression 
# This will use the mulit layer perceptron regression to construct a model on a training set, then perform analysis on the test data for performance evaluation.
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# 
# Optimize alpha
# 
# DXG - 2021-09-25

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix


# In[2]:


# User defined function for accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


# In[3]:


# Load dataset
#dataset = pd.read_csv('../featureSelectedDataset.csv')  
dataset = pd.read_csv('../LassoRegression/LassoForwardSelectionDataSetForModelling.csv')  
dataset.head()


# In[4]:


# Split the dataset into features and obs

forwardSelectionFeatures = pd.read_csv('../LassoRegression/ForwardSelectionVia5FoldCV.csv')["0"].tolist()
X = dataset[np.intersect1d(dataset.columns, forwardSelectionFeatures)]

#X = dataset.iloc[:,0:10]
y = dataset["Life expectancy at birth, total (years)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[32]:


# Fit the model and get accuracy
clf = MLPRegressor(solver='adam', activation='logistic', alpha=1e-2, hidden_layer_sizes=(10,5), random_state=1, max_iter=20000)

mlp = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[33]:


print('Accuracy training : {:.3f}'.format(mlp.score(X_train, y_train)))
print('Accuracy testing : {:.3f}'.format(mlp.score(X_test, y_test)))


# In[7]:


residuals = y_pred-y_test
residuals
residuals.shape, X_test.shape


# In[8]:


X_test.iloc[:,0]


# In[9]:


import matplotlib.pyplot as plt
import numpy as np
import os


# In[10]:


X_test.columns[0]


# In[11]:


def residual(x,y,title,folder):
    isExist = os.path.exists(folder)

    if not isExist:

      # Create a new directory because it does not exist 
      os.makedirs(folder)
    f = plt.figure()
    f = plt.scatter(x, y)
    plt.title(title)
    plt.savefig(folder + '/' + title + '.png', bbox_inches='tight', dpi=300)


# In[12]:


for index in range(0,10):
    residual(X_test.iloc[:,index], residuals,X_test.columns[index],'MLPRegression')


# In[ ]:




