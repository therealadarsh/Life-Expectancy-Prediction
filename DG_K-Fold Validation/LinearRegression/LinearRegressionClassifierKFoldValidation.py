#!/usr/bin/env python
# coding: utf-8

# ### Linear Regression Classifier 
# 
# DXG - 2021-10-10

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix


# In[4]:


# User defined function for accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


# In[5]:


# Load dataset
dataset = pd.read_csv('../../featureSelectedDataset.csv')  
dataset.head()


# In[6]:


# Split the dataset into features and obs
X = dataset.iloc[:,0:10]
y = dataset["Life expectancy at birth, total (years)"]


# In[7]:


# Perform cross validation 
from sklearn.model_selection import cross_val_score
# Fit the model
model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[20]:


# Feature selection
from sklearn.feature_selection import SequentialFeatureSelector

feature_names = X.columns
sfs_forward = SequentialFeatureSelector(model, n_features_to_select=5).fit(X,y)

sfs_backward = SequentialFeatureSelector(model, n_features_to_select=5, direction='backward').fit(X, y)


print("Features selected by forward sequential selection: " 
      f"{feature_names[sfs_forward.get_support()]}")
print("Features selected by backward sequential selection: "
      f"{feature_names[sfs_backward.get_support()]}")


# In[24]:


X5=X[feature_names[sfs_forward.get_support()]]
# Perform cross validation 
from sklearn.model_selection import cross_val_score
# Fit the model
model = LinearRegression()
cv_scores = cross_val_score(model, X5, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[27]:


# Feature selection
from sklearn.feature_selection import SequentialFeatureSelector

feature_names = X.columns
sfs_forward = SequentialFeatureSelector(model, n_features_to_select=3).fit(X,y)

sfs_backward = SequentialFeatureSelector(model, n_features_to_select=3, direction='backward').fit(X, y)


print("Features selected by forward sequential selection: " 
      f"{feature_names[sfs_forward.get_support()]}")
print("Features selected by backward sequential selection: "
      f"{feature_names[sfs_backward.get_support()]}")


# In[28]:


X3=X[feature_names[sfs_forward.get_support()]]
# Perform cross validation 
from sklearn.model_selection import cross_val_score
# Fit the model
model = LinearRegression()
cv_scores = cross_val_score(model, X3, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[31]:


X2=X[['Birth rate, crude (per 1,000 people)','CO2 emissions from solid fuel consumption (% of total)']]

# Perform cross validation 
from sklearn.model_selection import cross_val_score
# Fit the model
model = LinearRegression()
cv_scores = cross_val_score(model, X2, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[32]:


X1=X[['Birth rate, crude (per 1,000 people)']]

# Perform cross validation 
from sklearn.model_selection import cross_val_score
# Fit the model
model = LinearRegression()
cv_scores = cross_val_score(model, X1, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[ ]:




