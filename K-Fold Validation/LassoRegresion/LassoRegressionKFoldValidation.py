#!/usr/bin/env python
# coding: utf-8

# ### KNN Classifier 
# 
# DXG - 2021-10-10

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

from sklearn.metrics import confusion_matrix


# In[9]:


# User defined function for accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


# In[10]:


# Load dataset
dataset = pd.read_csv('../../featureSelectedDataset.csv')  
dataset.shape
dataset.head()


# In[11]:


# Split the dataset into features and obs
X = dataset.iloc[:,0:10]
y = dataset["Life expectancy at birth, total (years)"]


# In[14]:


# Perform cross validation 
from sklearn.model_selection import cross_val_score
# Define the model
model = Lasso(random_state=0, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{'alpha': alphas}]
n_folds = 10
cv_scores = cross_val_score(model, X, y, cv=n_folds)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[16]:


from sklearn.model_selection import GridSearchCV

#use gridsearch to test alphas
model_gscv = GridSearchCV(model, tuned_parameters, cv=n_folds, refit=False)
#fit model to data
model_gscv.fit(X, y)


# In[17]:


#check top performing n_neighbors value
model_gscv.best_params_


# In[18]:


#check mean score for the top performing value of n_neighbors
model_gscv.best_score_


# In[ ]:




