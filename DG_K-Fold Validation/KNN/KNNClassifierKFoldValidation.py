#!/usr/bin/env python
# coding: utf-8

# ### KNN Classifier 
# 
# DXG - 2021-10-10

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
#dataset = pd.read_csv('../../featureSelectedDataset.csv')  
dataset = pd.read_csv('../../LassoRegression/LassoForwardSelectionDataSetForModelling.csv')  
dataset.shape


# In[31]:


knnAll = KNeighborsClassifier(n_neighbors=5).fit(X.values,y.values)
preds=knnAll.predict(X.values)
print(confusion_matrix(y, preds))
print((2284+3448)/(2284+3448+296+231))


# In[43]:


2284+231


# In[41]:


df = pd.DataFrame(preds)
df[df[0] == False].count


# In[4]:


# Split the dataset into features and obs
#X = dataset.iloc[:,0:10]
forwardSelectionFeatures = pd.read_csv('../../LassoRegression/ForwardSelectionVia5FoldCV.csv')["0"].tolist()
X = dataset[np.intersect1d(dataset.columns, forwardSelectionFeatures)]
y = dataset["AboveAverageLifeExpectancyByYear"]


# In[13]:


# Perform cross validation for n=2
from sklearn.model_selection import cross_val_score
# Fit the model
knn = KNeighborsClassifier(n_neighbors=2)
cv_scores = cross_val_score(knn, X.values, y.values, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[5]:


# Perform cross validation for n=3
from sklearn.model_selection import cross_val_score
# Fit the model
knn = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(knn, X.values, y.values, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[6]:


# Perform cross validation for n=4
from sklearn.model_selection import cross_val_score
# Fit the model
knn = KNeighborsClassifier(n_neighbors=4)
cv_scores = cross_val_score(knn, X.values, y.values, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[7]:


# Perform cross validation for n=5
from sklearn.model_selection import cross_val_score
# Fit the model
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X.values, y.values, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[8]:


# Perform cross validation for n=6
from sklearn.model_selection import cross_val_score
# Fit the model
knn = KNeighborsClassifier(n_neighbors=6)
cv_scores = cross_val_score(knn, X.values, y.values, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[17]:


from sklearn.model_selection import GridSearchCV
#create new a knn model
knn_k = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {"n_neighbors": np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn_k, param_grid, cv=10)
#fit model to data
knn_gscv.fit(X.values, y.values)


# In[18]:


#check top performing n_neighbors value
knn_gscv.best_params_


# In[19]:


#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_


# In[20]:


sorted(knn_gscv.cv_results_.keys())


# In[21]:


df = pd.DataFrame(knn_gscv.cv_results_) 


# In[22]:


df


# In[ ]:




