#!/usr/bin/env python
# coding: utf-8

# ### Logisitic Regression Classifier 
# 
# DXG - 2021-10-10

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix


# In[3]:


# User defined function for accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


# In[5]:


# Load dataset
dataset = pd.read_csv('../../featureSelectedDataset.csv')  
dataset.shape


# In[6]:


# Split the dataset into features and obs
X = dataset.iloc[:,0:10]
y = dataset["AboveAverageLifeExpectancyByYear"]


# In[9]:


# Perform cross validation 
from sklearn.model_selection import cross_val_score
# Fit the model
model = LogisticRegression(random_state=0, max_iter=500)
cv_scores = cross_val_score(model, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

