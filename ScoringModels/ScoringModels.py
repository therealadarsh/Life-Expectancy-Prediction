#!/usr/bin/env python
# coding: utf-8

# ## Cross validation scores

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

# Regressors
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

seed =1


# In[2]:


df = pd.read_csv('../LassoRegression/LassoForwardSelectionDataSetForModelling.csv')
#lassoFeatures = pd.read_csv('top10predictorsLasso_cv.csv')['0'].tolist()
forwardSelectionFeatures = pd.read_csv('../LassoRegression/ForwardSelectionVia5FoldCV.csv')["0"].tolist()

y_qualitative = df['AboveAverageLifeExpectancyByYear']
y_quantitative = df['Life expectancy at birth, total (years)']
#xLasso = df[np.intersect1d(df.columns, lassoFeatures)]
x = df[np.intersect1d(df.columns, forwardSelectionFeatures)]

forwardSelectionFeatures


# In[3]:


allScores = pd.DataFrame(columns=['Type','Model', 'Score'])


# ### Classifier Models

# In[4]:


# KNN
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, x.values, y_qualitative, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
allScores.loc[allScores.shape[0]-1] = ['Classifier', 'KNN n = 5', np.mean(cv_scores)]


# In[5]:


# Logistic Regression
model = LogisticRegression(random_state=0, max_iter=500)
cv_scores = cross_val_score(model, x.values, y_qualitative, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
allScores.loc[allScores.shape[0]-1] = ['Classifier', 'Logistic Regression', np.mean(cv_scores)]


# In[6]:


# Multi Layer Perceptron Classifier
model = MLPClassifier(solver='adam', activation='tanh', alpha=1e-2, hidden_layer_sizes=(5,2), random_state=1, max_iter=20000)
cv_scores = cross_val_score(model, x.values, y_qualitative, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
allScores.loc[allScores.shape[0]-1] = ['Classifier', 'Multi Layer Perceptron', np.mean(cv_scores)]


# In[7]:


# Random Forest
model = RandomForestClassifier(
                      min_samples_leaf=50,
                      n_estimators=150,
                      bootstrap=True,
                      oob_score=True,
                      n_jobs=-1,
                      random_state=seed,
                      max_features='auto')
cv_scores = cross_val_score(model, x.values, y_qualitative, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
allScores.loc[allScores.shape[0]-1] = ['Classifier', 'Random Forest', np.mean(cv_scores)]


# ## Regession models

# In[8]:


# Multi Layer Perceptron Regression
model = MLPRegressor(solver='adam', activation='logistic', alpha=1e-2, hidden_layer_sizes=(10,5), random_state=1, max_iter=20000)
cv_scores = cross_val_score(model, x.values, y_quantitative, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
allScores.loc[allScores.shape[0]-1] = ['Regression', 'Multi Layer Perceptron', np.mean(cv_scores)]


# In[9]:


# KNN Regression
model = KNeighborsRegressor(n_neighbors=12)
cv_scores = cross_val_score(model, x.values, y_quantitative, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
allScores.loc[allScores.shape[0]-1] = ['Regression', 'KNN n = 12', np.mean(cv_scores)]


# In[10]:


# Multilinear Regression
model = LinearRegression()
cv_scores = cross_val_score(model, x.values, y_quantitative, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
allScores.loc[allScores.shape[0]-1] = ['Regression', 'Multilinear', np.mean(cv_scores)]


# In[13]:


# Random Forest
model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
cv_scores = cross_val_score(model, x.values, y_qualitative, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
allScores.loc[allScores.shape[0]-1] = ['Regression', 'Random Forest', np.mean(cv_scores)]


# In[14]:


display(allScores)


# In[ ]:


allScores.to_csv("allScores.csv",index=False )

