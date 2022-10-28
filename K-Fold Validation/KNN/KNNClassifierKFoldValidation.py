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


# In[4]:


# Split the dataset into features and obs
X = dataset.iloc[:,0:10]
y = dataset["AboveAverageLifeExpectancyByYear"]


# In[5]:


# Perform cross validation for n=3
from sklearn.model_selection import cross_val_score
# Fit the model
knn = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(knn, X.values, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[6]:


from sklearn.model_selection import GridSearchCV
#create new a knn model
knn_k = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {"n_neighbors": np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn_k, param_grid, cv=10)
#fit model to data
knn_gscv.fit(X.values, y)


# In[7]:


#check top performing n_neighbors value
knn_gscv.best_params_['n_neighbors']


# In[8]:


#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_


# In[9]:


#import seaborn as sns # for Data visualization

scores = knn_gscv.cv_results_['mean_test_score']
x = np.array(range(1,len(scores)+1))
plt.scatter(x, scores)
#plt.figure(figsize=(8,5))
#sns.scatterplot(x=x,y=scores)
plt.title("Cross Validation test Score vs n")
plt.xlabel("n")
plt.ylabel("Score")
plt.scatter(x=knn_gscv.best_params_['n_neighbors'], y=knn_gscv.best_score_, color='r')
plt.show()


# In[ ]:




